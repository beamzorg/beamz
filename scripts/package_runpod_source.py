"""Package current source, including uncommitted CUDA work, without pushing.

The archive contains an independent Git snapshot so benchmark provenance works
without carrying this worktree's absolute .git pointer or repository history.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT):
    return subprocess.check_output(
        ["git", "-c", "maintenance.auto=false", "-c", "gc.auto=0", *args], cwd=cwd
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    names = set(git("ls-files", "-z").decode().strip("\0").split("\0"))
    untracked = (
        git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    for name in untracked:
        path = Path(name)
        if (
            path.parts
            and path.parts[0] in {"beamz", "cuda", "scripts", "tests", "docker"}
            and path.suffix
            in {
                ".py",
                ".sh",
                ".md",
                ".toml",
                ".json",
                ".cu",
                ".cuh",
                ".h",
                ".cc",
                ".cpp",
            }
        ):
            names.add(name)
    names.update(
        {
            "docs/reviews/cuda-random-final-comparison-2026-09-18.md",
            "docs/reviews/rtx3090-2026-09-18-random-final/analysis.json",
        }
    )
    files = [ROOT / name for name in sorted(names) if name]
    files = [
        p for p in files if p.is_file() and not p.is_symlink() and p.suffix != ".so"
    ]
    manifest = dict(
        base_commit=git("rev-parse", "HEAD").decode().strip(),
        description="Uncommitted working source; independent snapshot Git history; no local binaries",
        files={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="beamz-runpod-") as tmp:
        snapshot = Path(tmp) / "beamz-h100"
        snapshot.mkdir()
        for source in files:
            target = snapshot / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (snapshot / "RUNPOD_SOURCE.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        git("init", "--quiet", "-b", "runpod-snapshot", cwd=snapshot)
        git("add", "--all", cwd=snapshot)
        git(
            "-c",
            "user.name=BeamZ source snapshot",
            "-c",
            "user.email=snapshot@localhost",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "RunPod snapshot of local CUDA branch; see RUNPOD_SOURCE.json",
            cwd=snapshot,
        )
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                archive.add(snapshot, arcname="beamz-h100")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    print(
        f"{output} ({output.stat().st_size / 1024**2:.1f} MiB, {len(files)} source files)"
    )
    print("SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
