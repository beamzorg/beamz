"""Derive neighbour routes and shared-family schedules from a cell trace.

This is a restricted schedule compiler, not a tracer for arbitrary Python. It
consumes the existing typed Yee/CPML DAG, keeps space symbolic, and selects a
CUDA communication/storage schedule. The first lowering spans a cache column
in private registers, exchanges X neighbours with full-warp shuffles and uses
shared memory only for Y neighbours. Independent H/E banks admit two barriers;
aliased banks require four. Neither the arithmetic nor global tile ownership
is changed. Source and observer effects remain in their original paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json

from scripts.cuda_space_time_ir import (
    FIELDS, IllegalSchedule, validate_operator, validate_stream_signature,
    yee_operator,
)
from scripts.cuda_stencil_trace import serialize_updates


@dataclass(frozen=True)
class ThreadMap:
    cache: tuple[int, int, int] = (8, 16, 32)
    threads: int = 512
    private_axis: int = 0
    shuffle_axis: int = 2
    z_groups: int = 1

    def validate(self):
        if (len(self.cache) != 3 or any(type(n) is not int or n < 2 for n in self.cache)
                or self.cache[2] != 32 or type(self.z_groups) is not int or self.z_groups < 1
                or self.cache[0] % self.z_groups or self.threads != self.cache[1] * 32 * self.z_groups
                or self.threads > 1024 or self.private_axis != 0 or self.shuffle_axis != 2):
            raise IllegalSchedule('Column lowering requires one complete X warp per Y row')
        if any(n & (n - 1) for n in self.cache):
            raise IllegalSchedule('First column lowering requires power-of-two cache extents')


@dataclass(frozen=True, order=True)
class RoutedRead:
    output: str
    source: str
    version: int
    offset: tuple[int, int, int]
    route: str


@dataclass(frozen=True)
class Effect:
    name: str
    phase: int
    role: str
    ordering: str


def derive_routes(operator=None, mapping=ThreadMap()):
    mapping.validate()
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    validate_operator(operator)
    validate_stream_signature(operator)
    specs = operator.specs
    routes = []
    for update in operator.updates:
        for read in sorted(update.reads):
            if specs[read.field].role == 'local-state':
                route = 'point-owned-global-recurrence'
            elif not any(read.offset):
                route = 'private-register'
            else:
                axes = [i for i, d in enumerate(read.offset) if d]
                if len(axes) != 1:
                    raise IllegalSchedule('Only axial field neighbours admitted')
                axis = axes[0]
                route = (('private-column-register' if mapping.z_groups == 1 else 'private-column-or-shared-z-edge') if axis == mapping.private_axis
                         else 'full-warp-shuffle' if axis == mapping.shuffle_axis
                         else 'shared-y-neighbour')
            routes.append(RoutedRead(update.output.field, read.field, read.version,
                                     read.offset, route))
    return tuple(sorted(set(routes)))


def compile_column(operator=None, mapping=ThreadMap(), *, separate_families=True,
                   requested_barriers=None):
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    routes = derive_routes(operator, mapping)
    shared = tuple(f for f in FIELDS if any(
        r.source == f and r.route == 'shared-y-neighbour' for r in routes))
    if shared != ('Hx', 'Hz', 'Ex', 'Ez'):
        raise IllegalSchedule('Generated two-component shared layout requires canonical Y curls')
    barriers = 2 if separate_families else 4
    if requested_barriers is not None and requested_barriers < barriers:
        raise IllegalSchedule('Aliased H/E banks need reader-retirement barriers')
    # Shared operations between block barriers must use disjoint physical banks.
    # Read+write inside a compute event is thread-private; CPML is point-owned.
    events = [('H.compute', 'read', 'E'), ('H.publish', 'write', 'H'),
              ('H.visible', 'barrier', ''), ('E.compute', 'read', 'H'),
              ('E.publish', 'write', 'E'), ('E.visible', 'barrier', '')]
    if not separate_families:
        events.insert(1, ('E.readers.retired', 'barrier', ''))
        events.insert(5, ('H.readers.retired', 'barrier', ''))
    pending = []
    checked = 0
    # Check two iterations, including the wrap from the first into the second.
    for _, kind, family in events * 2:
        if kind == 'barrier':
            pending.clear()
            continue
        bank = family if separate_families else 'aliased'
        for previous_kind, previous_bank in pending:
            if bank == previous_bank and 'write' in (kind, previous_kind):
                raise IllegalSchedule('Unretired interwarp shared-memory reader or writer')
            checked += 1
        pending.append((kind, bank))
    effects = [Effect('psi.' + t.name, t.phase, 'local-state',
                      'all component SSA reads precede the point-owned store')
               for t in operator.tensors if t.role == 'local-state']
    effects += [Effect('source.H', 0, 'source', 'finish before any H consumer'),
                Effect('source.E', 1, 'source', 'finish before any E consumer'),
                Effect('raw-observation.H', 0, 'observer', 'after sourced H value; before its next overwrite'),
                Effect('raw-observation.E', 1, 'observer', 'after sourced E value; before its next overwrite'),
                Effect('DFT', 2, 'ordered-reduction', 'existing chronological slab reduction'),
                Effect('trails', 2, 'publication', 'rank completion precedes downstream consumers'),
                Effect('final-state', 2, 'terminal', 'unique physical/version owner')]
    graph = serialize_updates(operator.updates)
    report = dict(schema='beamz.trace-column-schedule.v1', mapping=asdict(mapping),
                  owned_zyx=[n - 1 for n in mapping.cache], ownership_skew=[1, 1, 1],
                  routes=[asdict(r) for r in routes], shared_fields=tuple(f for f in FIELDS if f in shared or
                        (mapping.z_groups>1 and f[-1] in 'xy')),
                  shared_y_fields=shared,shared_z_fields=tuple(f for f in FIELDS if f[-1] in 'xy') if mapping.z_groups>1 else (),
                  separate_families=separate_families, barriers_per_step=barriers,
                  event_schedule=events, disjoint_shared_event_pairs=checked,
                  private_field_floats_per_thread=6 * mapping.cache[0] // mapping.z_groups,
                  shared_field_bytes=(4 if separate_families else 2) * 4 *
                                     mapping.cache[0] * mapping.cache[1] * mapping.cache[2]
                                     + (4 if separate_families else 2) * 4 * mapping.z_groups *
                                     (mapping.cache[1]-1)*(mapping.cache[2]-1) * (mapping.z_groups > 1),
                  shared_field_bytes_scope='Uncompacted Y planes and compact Z frontiers; lowering may pack and deduplicate further',
                  effects=[asdict(e) for e in effects],
                  cell_graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
                  restrictions=['Canonical axial Yee/point-local CPML only',
                                'Full-warp shuffles execute before owner predicates',
                                'Import new ghosts only after all old-E private reads',
                                'Native lowering selects certified NoIO IDs only',
                                'Sources, raw observers and DFT keep existing CUDA paths',
                                'Register counts are requirements, not ptxas measurements'])
    report['schedule_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return report


def prove_routes(mapping=ThreadMap(), *, steps=16):
    """Check every cache address against the independent physical-coordinate map.

    Slot identity is a bit-exact value token; no numerical equations are run.
    All ghost slots participate in shuffles, even when they own no output.
    """
    routes = derive_routes(mapping=mapping)
    cache = mapping.cache
    checks = 0
    for n in range(1, steps + 1):
        low = -n - 1
        heads = tuple(low % c for c in cache)
        for slot in itertools.product(*(range(c) for c in cache)):
            if any(a == h for a, h in zip(slot, heads)):
                continue
            physical = tuple(low + (a - h) % c for a, h, c in zip(slot, heads, cache))
            for read in routes:
                if not any(read.offset):
                    continue
                source_slot = tuple((a + d) % c for a, d, c in zip(slot, read.offset, cache))
                # H reads the previous version before ghost imports; E reads
                # the current version after H publication. The rotating slot
                # has a different physical interpretation at those versions.
                source_low = low + 1 - read.version
                source_heads = tuple(source_low % c for c in cache)
                source_physical = tuple(source_low + (a - h) % c
                                        for a, h, c in zip(source_slot, source_heads, cache))
                expected = tuple(a + d for a, d in zip(physical, read.offset))
                if source_physical != expected:
                    raise IllegalSchedule('Neighbour route crossed an unimported cache boundary')
                if read.route == 'private-column-register' and source_slot[1:] != slot[1:]:
                    raise IllegalSchedule('Private read changes thread owner')
                if read.route == 'private-column-or-shared-z-edge' and source_slot[1:] != slot[1:]:
                    raise IllegalSchedule('Z-edge route changes an unrelated thread coordinate')
                if read.route == 'full-warp-shuffle' and source_slot[:2] != slot[:2]:
                    raise IllegalSchedule('Shuffle read changes warp owner')
                checks += 1
    return dict(steps=steps, cache_zyx=cache, checked_routed_neighbours=checks,
                includes_all_ghost_shuffle_providers=True, result='pass')


def compact_publication(slot, version, family, cache=(8, 16, 32)):
    """Rank a shared-Y provider, or return None for a dead shared write.

    H is consumed at this version with negative Y offsets. E is consumed in
    the next version with positive Y offsets. Their minimal provider boxes
    have different translations, although both contain BZ*BY*BX points.
    """
    if family not in ('H', 'E'):
        raise IllegalSchedule('Shared publication needs an H/E family')
    owned = tuple(n - 1 for n in cache)
    dz, dy, dx = ((a + version + 1) % c for a, c in zip(slot, cache))
    bz, by, bx = owned
    if family == 'H':
        if dz == 0 or dy == by or dx == 0:
            return None
        z, y, x = dz - 1, dy, dx - 1
    else:
        if dz == bz or dy == 0 or dx == bx:
            return None
        z, y, x = dz, dy - 1, dx
    return (z * by + y) * bx + x


def prove_compact_providers(mapping=ThreadMap(), *, steps=16):
    """Inverse provider boxes are bijective and cover every traced shared read."""
    mapping.validate()
    cache = mapping.cache
    owned = tuple(n - 1 for n in cache)
    bz, by, bx = owned
    points = bz * by * bx
    routes = tuple(r for r in derive_routes(mapping=mapping) if r.route == 'shared-y-neighbour')
    checks, writes = 0, 0
    for n in range(1, steps + 1):
        for family, version in (('E', n - 1), ('H', n)):
            addresses = [compact_publication(slot, version, family, cache)
                         for slot in itertools.product(*(range(c) for c in cache))]
            live = [index for index in addresses if index is not None]
            if len(live) != points or set(live) != set(range(points)):
                raise IllegalSchedule('Compact provider layout has an alias or a hole')
            writes += len(live)
        for slot in itertools.product(*(range(c) for c in cache)):
            dz, dy, dx = ((a + n + 1) % c for a, c in zip(slot, cache))
            if not dz or not dy or not dx:
                continue
            expected_index = ((dz - 1) * by + dy - 1) * bx + dx - 1
            for read in routes:
                provider = tuple((a + d) % c for a, d, c in zip(slot, read.offset, cache))
                version = n - 1 + read.version
                actual = compact_publication(provider, version, read.source[0], cache)
                if actual != expected_index:
                    raise IllegalSchedule('Compact provider does not match traced consumer')
                checks += 1
    return dict(result='pass', steps=steps, words_per_component=points,
                checked_shared_reads=checks, checked_unique_provider_writes=writes,
                H_relative_provider_box=[[1, bz + 1], [0, by], [1, bx + 1]],
                E_relative_provider_box=[[0, bz], [1, by + 1], [0, bx]],
                bounds='half-open; relative to this version low ghost',
                note='Only shared-Y providers; private/shuffle reads retain all ghost registers')


def compact_z_publication(slot, version, family, mapping=ThreadMap(threads=1024, z_groups=2)):
    """Only the two tangential Z components at private-chunk boundaries live."""
    mapping.validate()
    z, y, x = slot
    cz, cy, cx = mapping.cache
    by, bx = cy-1, cx-1
    chunk = cz // mapping.z_groups
    dy, dx = (y+version+1) % cy, (x+version+1) % cx
    if family == 'H':
        if z % chunk != chunk-1 or not dy or not dx:
            return None
        yindex, xindex = dy-1, dx-1
    elif family == 'E':
        if z % chunk or dy == by or dx == bx:
            return None
        yindex, xindex = dy, dx
    else:
        raise IllegalSchedule('Shared Z publication needs an H/E family')
    return (z//chunk*by+yindex)*bx+xindex


def prove_z_providers(mapping=ThreadMap(threads=1024, z_groups=2), *, steps=16):
    mapping.validate()
    cz, cy, cx = mapping.cache
    chunk = cz // mapping.z_groups
    by, bx = cy-1, cx-1
    routes = [r for r in derive_routes(mapping=mapping) if r.route == 'private-column-or-shared-z-edge']
    checks = 0
    for n in range(1, steps+1):
        for family, version in (('E', n-1), ('H', n)):
            live = [compact_z_publication(slot, version, family, mapping)
                    for slot in itertools.product(*(range(c) for c in mapping.cache))]
            live = [v for v in live if v is not None]
            if len(live) != mapping.z_groups*by*bx or set(live) != set(range(len(live))):
                raise IllegalSchedule('Shared Z edge publication aliases or has holes')
        for slot in itertools.product(*(range(c) for c in mapping.cache)):
            dz, dy, dx = ((a+n+1) % c for a, c in zip(slot, mapping.cache))
            if not dz or not dy or not dx:
                continue
            for read in routes:
                znext = (slot[0]+read.offset[0]) % cz
                if znext//chunk == slot[0]//chunk:
                    continue
                if read.source[-1] not in 'xy':
                    raise IllegalSchedule('Z-normal component cannot use the edge layout')
                expected = (znext//chunk*by+dy-1)*bx+dx-1
                actual = compact_z_publication((znext, slot[1], slot[2]), n-1+read.version,
                                               read.source[0], mapping)
                if actual != expected:
                    raise IllegalSchedule('Shared Z edge provider misses its traced consumer')
                checks += 1
    return dict(result='pass', steps=steps, z_groups=mapping.z_groups,
                private_z_chunk=chunk, checked_z_edge_reads=checks,
                shared_z_words_per_component=mapping.z_groups*by*bx)


def deduplicated_z_publication(slot, version, family, component,
                               mapping=ThreadMap(threads=1024, z_groups=2)):
    """Return ('Y'/'Z', word) for a live Z-edge value; reuse X components in Y.

    The normal-Y component is not present in Y buffers. The X component usually
    is; only its missing Z plane and missing Y row require extra words.
    """
    regular = compact_z_publication(slot, version, family, mapping)
    if regular is None:
        return None
    if component not in (0, 1):
        raise IllegalSchedule('Only X/Y components have Z neighbours')
    cz, cy, cx = mapping.cache
    bz, by, bx = cz-1, cy-1, cx-1
    groups = mapping.z_groups
    if component == 1:
        return ('Z', regular)
    ybuffer = compact_publication(slot, version, family, mapping.cache)
    if ybuffer is not None:
        return ('Y', ybuffer)
    z, y, x = slot
    dz, dy, dx = ((a+version+1) % c for a, c in zip(slot, mapping.cache))
    base = groups*by*bx
    ghost = dz == (bz if family == 'E' else 0)
    if ghost:
        index = dy*bx+dx if family == 'E' else (dy-1)*bx+dx-1
    else:
        index = by*bx+(z//(cz//groups))*bx+(dx if family == 'E' else dx-1)
    return ('Z', base+index)


def prove_deduplicated_z(mapping=ThreadMap(threads=1024,z_groups=2), *, steps=16):
    mapping.validate()
    if mapping.z_groups < 2:
        raise IllegalSchedule('Deduplicated frontier lowering requires multiple Z groups')
    cz, cy, cx = mapping.cache
    by, bx = cy-1, cx-1
    words = mapping.z_groups*by*bx+by*bx+mapping.z_groups*bx
    checks, saved = 0, 0
    for version in range(steps+1):
        for family in ('H', 'E'):
            writers = {}
            for component in (0, 1):
                for slot in itertools.product(*(range(c) for c in mapping.cache)):
                    address = deduplicated_z_publication(slot,version,family,component,mapping)
                    if address is None:
                        continue
                    bank, word = address
                    if bank == 'Z':
                        if not 0 <= word < words:
                            raise IllegalSchedule('Deduplicated frontier out of allocation')
                        if word in writers and writers[word] != (component,slot):
                            raise IllegalSchedule('Deduplicated frontier has aliased live writers')
                        writers[word] = (component,slot)
                    else:
                        if component != 0 or compact_publication(slot,version,family,mapping.cache) != word:
                            raise IllegalSchedule('Invalid reuse of a Y publication')
                        saved += 1
                    checks += 1
    return dict(result='pass', checked_z_publications=checks, reused_y_publications=saved,
                allocated_z_words_per_family=words,
                prior_z_words_per_family=2*mapping.z_groups*by*bx,
                extra_X_storage='One missing Z plane plus two missing Y rows; live-writer alias check')


if __name__ == '__main__':
    print(json.dumps(dict(schedule=compile_column(), routing_proof=prove_routes()), indent=2))
