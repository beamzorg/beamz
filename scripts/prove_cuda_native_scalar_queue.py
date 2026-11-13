"""Independent fixed-scalar queue proof for the native asymmetric Z loop.

This checks storage/version routing, not floating-point arithmetic or CUDA
compiler register allocation. Two private six-slot families rotate after EVERY
physical j, including ghost/invalid lanes. Six rotations restore orientation
before the unchanged import and publication loops. All X shuffles participate.
"""

from __future__ import annotations

import json


def prove(depth=16, *, skip_excluded_rotation=False, wrong_previous=False,
          omit_ghost_shuffle=False):
    cz, cy, cx = 6, 16, 32
    checks = 0
    for sy in range(cy):
        # Tokens encode physical slot, lane, family, and last update version.
        state = [[[((family, 0, j, sy, sx)) for sx in range(cx)]
                  for j in range(cz)] for family in range(2)]
        for version in range(1, depth + 1):
            head, yh, xh = ((-version - 1) % size for size in (cz, cy, cx))
            for phase in (0, 1):
                before = [[list(row) for row in family] for family in state]
                # Each thread has fixed named scalar positions q0..q5. The
                # logical rotation is an assignment sequence, never indexing
                # a dynamically addressed register array in the CUDA lowering.
                queues = [[[(j, state[f][j][sx]) for j in range(cz)]
                           for sx in range(cx)] for f in range(2)]
                expected = [[list(row) for row in family] for family in state]
                for j in range(cz):
                    for family in range(2):
                        for sx in range(cx):
                            for k, (slot, _) in enumerate(queues[family][sx]):
                                assert slot == (j + k) % cz, "Scalar queue lost physical orientation"
                    participants = set(range(cx)) - ({xh} if omit_ghost_shuffle else set())
                    for sx in range(cx):
                        owned = j != head and sy != yh and sx != xh
                        if not owned:
                            continue
                        source_lane = (sx + (1 if phase == 0 else -1)) % cx
                        assert source_lane in participants, "Owned X derivative needs excluded ghost lane"
                        opposite = 1 - phase
                        # The private y component supplies two tangential
                        # field outputs: center/+Z/+X for H and center/-Z/-X
                        # for E. Old own y supplies the third field output.
                        z_slot = 1 if phase == 0 else (1 if wrong_previous else cz - 1)
                        actual = (queues[phase][sx][0][1],
                                  queues[opposite][sx][0][1],
                                  queues[opposite][sx][z_slot][1],
                                  queues[opposite][source_lane][0][1])
                        wanted = (before[phase][j][sx], before[opposite][j][sx],
                                  before[opposite][(j + (1 if phase == 0 else -1)) % cz][sx],
                                  before[opposite][j][source_lane])
                        assert actual == wanted, "Point callback read wrong slot/version"
                        token = (phase, version, j, sy, sx)
                        queues[phase][sx][0] = (j, token)
                        expected[phase][j][sx] = token
                        checks += 4
                    # Rotation stays outside ownership/support predicates.
                    for family in range(2):
                        for sx in range(cx):
                            if skip_excluded_rotation and (j == head or sy == yh or sx == xh):
                                continue
                            q = queues[family][sx]
                            queues[family][sx] = q[1:] + q[:1]
                for family in range(2):
                    for sx in range(cx):
                        for j, (slot, token) in enumerate(queues[family][sx]):
                            assert slot == j, "Six rotations failed to restore physical orientation"
                            assert token == expected[family][j][sx]
                            state[family][j][sx] = token
                if phase == 0:
                    # Existing B1/import/B2 installs both CURRENT families in
                    # the low union. It occurs with original queue orientation.
                    for j in range(cz):
                        for sx in range(cx):
                            if j == head or sy == yh or sx == xh:
                                for family in range(2):
                                    state[family][j][sx] = (family, version, j, sy, sx)
    return {"depth": depth, "private_floats_per_thread": 12,
            "point_body_copies_per_phase": 1, "verified_private_operands": checks,
            "physical_slots": cz, "result": "pass",
            "limits": "Storage-routing proof only; no arithmetic, ptxas, or performance claim"}


def main():
    cases = [prove(n) for n in (4, 8, 16, 32, 64)]
    rejected = []
    for option in ("skip_excluded_rotation", "wrong_previous", "omit_ghost_shuffle"):
        try:
            prove(**{option: True})
        except AssertionError:
            rejected.append(option)
        else:
            raise AssertionError(f"Negative control escaped: {option}")
    print(json.dumps({"cases": cases, "negative_controls_rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
