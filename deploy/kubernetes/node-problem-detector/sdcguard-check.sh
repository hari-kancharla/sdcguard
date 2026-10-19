#!/bin/sh
# node-problem-detector custom plugin: exit 0 = OK, 1 = a GPU computed wrong results, 2 = unknown.
# Reads the result of the node's last screen (screen-job.yaml) instead of screening: NPD runs this on a
# timer, possibly while jobs use the GPUs. Results older than 36 hours count as unknown.
exec sdcguard nhc --cached --out "${SDCGUARD_OUT:-/var/lib/sdcguard/screen}"
