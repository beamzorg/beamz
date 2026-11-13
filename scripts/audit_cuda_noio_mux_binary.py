"""Read-only CUDA object audit: text sizes, front-end locals and SASS spills.

Runs cuobjdump only. Never invokes nvcc, ptxas, a CUDA API or GPU execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sections(path):
    data = Path(path).read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected ELF64 little-endian CUDA object")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    entsize, count, strings = struct.unpack_from("<HHH", data, 58)
    rows = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * entsize) for i in range(count)]
    string_row = rows[strings]
    names = data[string_row[4]:string_row[4] + string_row[5]]
    return {names[r[0]:names.index(b"\0", r[0])].decode(): r[5] for r in rows}


def ptx_entries(text):
    result = {}
    for match in re.finditer(r"\.entry\s+(\S+)\s*\(", text):
        name = match.group(1)
        if "AsymmetricNoIOUniform1024" not in name:
            continue
        begin = text.index("{", match.end())
        level, end = 1, begin + 1
        while level:
            level += (text[end] == "{") - (text[end] == "}")
            end += 1
        body = text[begin:end]
        registers = re.findall(r"\.reg\s+(\.\w+)\s+([^;]+);", body)
        result[name] = {"PTX_text_bytes": end - match.start(),
                        "PTX_register_declarations": registers,
                        "PTX_local_declarations": re.findall(r"[^\n]*\.local[^\n]*", body),
                        "PTX_local_load_instructions": len(re.findall(r"\bld\.local\b", body)),
                        "PTX_local_store_instructions": len(re.findall(r"\bst\.local\b", body)),
                        "PTX_local_address_conversion_instructions": len(re.findall(r"\bcvta\.local\b", body)),
                        "PTX_dynamic_array_declarations": re.findall(r"[^\n]*\[[0-9]+\][^\n]*;", body),
                        "PTX_body_sha256": hashlib.sha256(body.encode()).hexdigest()}
    return result


def audit(label, obj, output, tool):
    obj, output = Path(obj).resolve(), Path(output).resolve()
    directory = output / label
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(tool), "--dump-ptx", str(obj)]
    ptx_path = directory / "embedded-pre-ptxas.ptx"
    with ptx_path.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.PIPE, text=True)
    subprocess.run([str(tool), "--extract-elf", "all", str(obj)], cwd=directory, check=True, capture_output=True, text=True)
    cubins = sorted(directory.glob("*.cubin"))
    if len(cubins) != 1:
        raise ValueError("Exactly one SM86 moving cubin required")
    cubin = cubins[0]
    entries = ptx_entries(ptx_path.read_text())
    text_sections = sections(cubin)
    resources = subprocess.run([str(tool), "--dump-resource-usage", str(obj)], check=True, capture_output=True, text=True).stdout
    (directory / "resources.txt").write_text(resources)
    for name, data in entries.items():
        data["SM86_text_section_bytes"] = text_sections[".text." + name]
        sass = subprocess.run([str(tool), "--dump-sass", "--function", name, str(obj)], check=True, capture_output=True, text=True).stdout
        (directory / (name + ".sass")).write_text(sass)
        data["SASS_LDL_instructions"] = len(re.findall(r"\bLDL(?:\.|\s)", sass))
        data["SASS_STL_instructions"] = len(re.findall(r"\bSTL(?:\.|\s)", sass))
        where = resources.index("Function " + name + ":")
        data["actual_resources"] = resources[where:resources.index("\n", resources.index("REG:", where))].strip()
    return {"object": str(obj), "object_sha256": digest(obj), "cubin_sha256": digest(cubin),
            "embedded_PTX_sha256": digest(ptx_path), "entries": entries,
            "scope": "Existing binary inspection only; retained compute86 PTX is pre-ptxas virtual-register code, SASS is actual SM86 machine code"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cuobjdump", default="/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/cuobjdump", type=Path)
    a = p.parse_args()
    report = {label: audit(label, path, a.output, a.cuobjdump) for label, path in (("baseline", a.baseline), ("candidate", a.candidate))}
    report.update(schema="beamz.noio-mux-existing-binary-audit.v1", GPU_executed=False, CUDA_compilation_executed=False,
                  producer_sha256=digest(__file__))
    comparisons = []
    for name, candidate in report["candidate"]["entries"].items():
        baseline = report["baseline"]["entries"][name]
        comparisons.append({"entry": name, "baseline_text_bytes": baseline["SM86_text_section_bytes"],
                            "candidate_text_bytes": candidate["SM86_text_section_bytes"],
                            "text_reduction_factor": baseline["SM86_text_section_bytes"] / candidate["SM86_text_section_bytes"],
                            "candidate_frontend_local_declarations": candidate["PTX_local_declarations"],
                            "candidate_actual_resources": candidate["actual_resources"]})
    report["comparison"] = comparisons
    (a.output / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
