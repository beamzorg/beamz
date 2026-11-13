"""Supplementary exact cross-warp H/E packet publication and cycle reuse proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def physical_witness():
    """Keys1/45 exchange producer/consumer warps at the actual n16 heads."""
    witness = []
    n, hy, hx = 16, 15, 15
    for key in (1, 45):
        dz, dy = 1 + key // 15, 1 + key % 15
        consumer = ((hy + dy) % 16) * 32 + hx
        witness.append({"key": key, "version": n, "logical_dzdy": [dz, dy],
                        "producer_thread": key, "producer_warp": key // 32,
                        "consumer_thread": consumer, "consumer_warp": consumer // 32})
    assert [(w["producer_warp"], w["consumer_warp"]) for w in witness] == [(0, 1), (1, 0)]
    return witness


def prove(*, omit_h_retirement=False, omit_e_publication=False, omit_next_retirement=False):
    queues = (("read_H1", "write_E0", "read_E1", "write_Hnext0"),
              ("read_H0", "write_E1", "read_E0", "write_Hnext1"))
    completed = 0

    def visit(indices, stage, history):
        nonlocal completed
        if indices == (4, 4):
            completed += 1
            return
        for worker in range(2):
            index = indices[worker]
            if index == 4:
                continue
            action = queues[worker][index]
            if action.startswith("write_E") and min(indices) < 1 and not omit_h_retirement:
                continue
            if action.startswith("read_E") and min(indices) < 2 and not omit_e_publication:
                continue
            if action.startswith("write_Hnext") and min(indices) < 3 and not omit_next_retirement:
                continue
            key = int(action[-1])
            updated = list(stage)
            if action.startswith("read"):
                expected = ("H", 1) if action.startswith("read_H") else ("E", 1)
                if stage[key] != expected:
                    raise AssertionError(f"Cross-warp packet lifetime failure: {history + [(worker, action)]}")
            else:
                updated[key] = ("E", 1) if action.startswith("write_E") else ("H", 2)
            next_indices = list(indices)
            next_indices[worker] += 1
            visit(tuple(next_indices), tuple(updated), history + [(worker, action)])

    visit((0, 0), (("H", 1), ("H", 1)), [])
    return {"legal_interleavings": completed, "result": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"scope": __doc__, "physical_witness": physical_witness(), "positive": prove()}
    rejected = []
    for option in ("omit_h_retirement", "omit_e_publication", "omit_next_retirement"):
        try:
            prove(**{option: True})
        except AssertionError:
            rejected.append(option)
        else:
            raise AssertionError(f"Negative escaped: {option}")
    result["negative_controls_rejected"] = rejected
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
