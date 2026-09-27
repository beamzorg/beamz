import datetime
import json
import pathlib
import time

out = pathlib.Path("/workspace/evidence/host-telemetry.jsonl")
while True:
    rows = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "/workspace/fixed/scripts/" not in command or not command.startswith(
                "/workspace/venv/bin/python "
            ):
                continue
            status = (entry / "status").read_text().splitlines()
            rows.append(
                dict(
                    pid=int(entry.name),
                    command=command,
                    memory={
                        line.split(":")[0]: line.split(":")[1].strip()
                        for line in status
                        if line.startswith(("VmRSS:", "VmHWM:", "VmPeak:"))
                    },
                )
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    with out.open("a") as f:
        f.write(
            json.dumps(
                dict(
                    time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    cgroup_bytes=int(
                        pathlib.Path("/sys/fs/cgroup/memory.current").read_text()
                    ),
                    processes=rows,
                )
            )
            + "\n"
        )
    time.sleep(5)
