#!/bin/bash
while pgrep -f '^/workspace/venv/bin/python /workspace/eight.py$' > /dev/null; do sleep 5; done
if grep -q EIGHT_TARGET_COMPLETE /workspace/evidence/runner.log; then
 /workspace/venv/bin/python /workspace/after.py > /workspace/evidence/after-runner.log 2>&1
fi
