#!/bin/sh
# Wait for the tonight.py runner to exit, then start after.py.
# stdout is appended to runs/tonight.out so the existing stage monitor sees it.
PY="/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe"
PID="$1"
while [ "$(powershell -NoProfile -Command "if (Get-Process -Id $PID -EA SilentlyContinue) {'ALIVE'} else {'GONE'}" | tr -d '\r')" = "ALIVE" ]; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] tonight.py (pid $PID) exited; starting after.py" >> runs/tonight.out
"$PY" scripts/after.py 6.0 >> runs/tonight.out 2>&1
echo "[$(date +%H:%M:%S)] after.py finished rc=$?" >> runs/tonight.out
