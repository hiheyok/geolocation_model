#!/bin/sh
# Start the pyramid cache once the offline block's last stage is marked done.
PY="/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe"
while [ ! -f runs/marks/kv_screen.done ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] offline block done; starting pyramid_cache" >> runs/tonight.out
OSV_RELEASE=s10 "$PY" scripts/pyramid_cache.py --data E:/data/kartaview_hr \
    --out pyr33 --batch 24 >> runs/logs/pyr33.log 2>&1
echo "[$(date +%H:%M:%S)] pyramid_cache rc=$?" >> runs/tonight.out
