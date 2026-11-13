"""Synthesize affine moving/diamond ownership from symbolic cell dependencies.

For each transformed coordinate, solve integer difference constraints for the
minimum time skew and per-component retiming. Coordinate rows may have negative
coefficients: the rotated XY example therefore needs nontrivial field retiming.
It is a CPU legality model, not an executable DiamondCandy CUDA backend.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path

from scripts.cuda_space_time_ir import IllegalSchedule, validate_operator, yee_operator
from scripts.prove_cuda_systolic_schedule import component_shapes

BOX_BASIS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
DIAMOND_XY_BASIS = ((1, 0, 0), (0, 1, 1), (0, -1, 1))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b, strict=True))


def determinant(m):
    a, b, c = m
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
            -a[1]*(b[0]*c[2]-b[2]*c[0])
            +a[2]*(b[0]*c[1]-b[1]*c[0]))


def dependence_edges(operator, row, speed):
    edges = []
    specs = operator.specs
    for update in operator.updates:
        for read in update.reads:
            dt = update.output.version-read.version
            edges.append((read.field, update.output.field, dot(row, read.offset)-speed*dt))
    # Local CPML state and its primary field must retain the same owner. This
    # permits the existing component SSA group to update mutable psi safely.
    for tensor in operator.tensors:
        if tensor.role == 'local-state':
            stem = tensor.name.removeprefix('psi_')
            primary = stem[0]+'xyz'[int(stem[1:])//2]
            if primary not in specs:
                raise IllegalSchedule('CPML state has no canonical primary field')
            edges += [(tensor.name, primary, 0), (primary, tensor.name, 0)]
    return tuple(edges)


def solve_retiming(operator, row, speed):
    """Longest-path relaxation; a positive cycle rejects this time skew."""
    values = {t.name:0 for t in operator.tensors}
    edges = dependence_edges(operator, row, speed)
    for _ in range(len(values)):
        changed = False
        for source, target, distance in edges:
            required = values[source]+distance
            if values[target] < required:
                values[target] = required
                changed = True
        if not changed:
            low = min(values.values())
            return {name:value-low for name, value in sorted(values.items())}
    raise IllegalSchedule('Positive dependency cycle for this spatial row/time skew')


def verify_transform(operator, basis, speeds, retiming):
    if (len(basis) != 3 or any(len(row) != 3 or any(type(x) is not int for x in row) for row in basis)
            or not determinant(basis) or len(speeds) != 3
            or any(type(v) is not int or v < 0 for v in speeds)):
        raise IllegalSchedule('Nonsingular integer spatial basis and nonnegative skew required')
    if set(retiming) != set(operator.specs) or any(len(r) != 3 or any(type(v) is not int for v in r) for r in retiming.values()):
        raise IllegalSchedule('Every evolving tensor needs three retiming offsets')
    for axis, (row, speed) in enumerate(zip(basis, speeds)):
        for source, target, distance in dependence_edges(operator, row, speed):
            if retiming[target][axis] < retiming[source][axis]+distance:
                raise IllegalSchedule(f'Future-owner dependence: {source}->{target}, coordinate {axis}')


def synthesize(basis=BOX_BASIS, *, operator=None, maximum_speed=4):
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    validate_operator(operator)
    if len(basis) != 3 or any(len(row) != 3 for row in basis) or not determinant(basis):
        raise IllegalSchedule('Nonsingular three-dimensional basis required')
    columns, speeds, rejected = [], [], []
    for axis, row in enumerate(basis):
        for speed in range(maximum_speed+1):
            try:
                offsets = solve_retiming(operator, row, speed)
            except IllegalSchedule:
                rejected.append(dict(axis=axis, speed=speed, reason='positive dependency cycle'))
                continue
            columns.append(offsets)
            speeds.append(speed)
            break
        else:
            raise IllegalSchedule('No admitted time skew within search bound')
    retiming = {t.name:tuple(column[t.name] for column in columns) for t in operator.tensors}
    verify_transform(operator, basis, speeds, retiming)
    return dict(schema='beamz.trace-affine-ownership.v1', spatial_basis=basis,
                determinant=determinant(basis), time_skew=speeds,
                tensor_retiming=retiming, rejected_smaller_skews=rejected,
                owner_rule='floor((A @ physical_position + skew*version + tensor_retiming)/tile)',
                rank_rule='sum(owner_coordinates); increasing local version then original phase order',
                parity_note='Non-unit determinant introduces lattice parity constraints; no fictitious cells may execute',
                GPU_executed=False, scope='Symbolic legality and CPU ownership proof, no CUDA performance claim')


def owner(plan, field, version, position, tile):
    return tuple((dot(row, position)+speed*version+r)//width
                 for row, speed, r, width in zip(plan['spatial_basis'], plan['time_skew'],
                    plan['tensor_retiming'][field], tile, strict=True))


def prove(plan, *, shape=(4, 5, 6), tile=(3, 5, 5), steps=4, drop_publication=False):
    operator = yee_operator(cpml_axes=(0, 1, 2))
    verify_transform(operator, plan['spatial_basis'], plan['time_skew'], plan['tensor_retiming'])
    if len(tile) != 3 or min(*shape, *tile, steps) < 1:
        raise IllegalSchedule('Positive dimensions, tile widths and timesteps required')
    shapes = component_shapes(shape)
    fields = tuple(t.name for t in operator.tensors)
    by_owner = defaultdict(list)
    # Bind spatial cells only in this finite independent checker; the compiler
    # itself never expands the simulation domain.
    for version in range(1, steps+1):
        for order, update in enumerate(operator.updates):
            field = update.output.field
            for position in itertools.product(*(range(n) for n in shapes[field])):
                key = owner(plan, field, version, position, tile)
                by_owner[key].append((version, order, field, position))
    produced, required = set(), set()
    stats = Counter()
    def valid(field, point):
        return all(0 <= a < n for a, n in zip(point, shapes[field]))
    for key in sorted(by_owner, key=lambda k:(sum(k), k)):
        for version, order, field, position in sorted(by_owner[key]):
            output = (field, version, position)
            if output in produced:
                raise IllegalSchedule('Duplicate physical/version ownership')
            update = operator.updates[order]
            for read in update.reads:
                q = tuple(a+d for a, d in zip(position, read.offset))
                if not valid(read.field, q):
                    stats['physical_wall_reads'] += 1
                    continue
                source_version = version-1+read.version
                source = (read.field, source_version, q)
                upstream = owner(plan, read.field, source_version, q, tile)
                if any(a > b for a, b in zip(upstream, key)):
                    raise IllegalSchedule('Producer lies in a future owner')
                if source_version and source not in produced:
                    raise IllegalSchedule('Consumer executes before its producer')
                if upstream != key:
                    required.add((source, key))
                    stats['cross_owner_read_operands'] += 1
                    stats['maximum_cross_owner_rank_distance'] = max(
                        stats['maximum_cross_owner_rank_distance'], sum(key)-sum(upstream))
                stats['valid_read_operands'] += 1
            produced.add(output)
    expected = steps*sum(math.prod(shapes[f]) for f in fields)
    if len(produced) != expected:
        raise IllegalSchedule('Missing physical/version updates')
    inverse = defaultdict(set)
    for update in operator.updates:
        for read in update.reads:
            inverse[read.field].add((update.output.field, 1-read.version,
                                     tuple(-d for d in read.offset)))
    initial = {(f, 0, p) for f in fields
               for p in itertools.product(*(range(n) for n in shapes[f]))}
    outgoing = set()
    for source in produced | initial:
        field, version, position = source
        upstream = owner(plan, field, version, position, tile)
        for target_field, dt, offset in inverse[field]:
            target_version = version+dt
            q = tuple(a+d for a, d in zip(position, offset))
            if not 1 <= target_version <= steps or not valid(target_field, q):
                continue
            downstream = owner(plan, target_field, target_version, q, tile)
            if downstream != upstream:
                outgoing.add((source, downstream))
    if drop_publication:
        outgoing.remove(next(iter(outgoing)))
    if outgoing != required:
        raise IllegalSchedule('Inverse publication set differs from consumer requirements')
    field_values = {s for s, _ in outgoing if not s[0].startswith('psi')}
    return dict(shape=shape, tile=tile, steps=steps, tensors=len(fields),
                unique_updates=len(produced), executed_over_useful_updates=1,
                owners=len(by_owner), publications=len(outgoing),
                unique_field_boundary_values=len(field_values),
                maximum_owner_updates=max(map(len, by_owner.values())),
                effect_requirements_for_lowering='Local CPML tied to primary field; source finish precedes publication; deferred DFT needs independent raw staging after all ranks',
                scope='CPU logical dependency/ownership proof; no in-place pointer, CUDA visibility or ordered DFT execution proof',
                result='pass', **stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    results = []
    for basis in (BOX_BASIS, DIAMOND_XY_BASIS):
        plan = synthesize(basis)
        cases = [prove(plan, shape=shape, tile=tile, steps=steps)
                 for shape, tile in (((4, 5, 6), (3, 5, 5)), ((3, 4, 3), (2, 3, 4)))
                 for steps in (4, 8)]
        results.append(dict(plan=plan, cases=cases))
    result = dict(schema='beamz.trace-affine-ownership-proof.v1', cases=results,
                  GPU_executed=False, note='Rotated example is diamond-inspired; not the published DiamondCandy algorithm')
    text = json.dumps(result, indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
