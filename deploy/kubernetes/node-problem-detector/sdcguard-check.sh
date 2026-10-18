#!/bin/sh
# node-problem-detector custom plugin: exit 0 = OK, 1 = a GPU computed wrong results, 2 = unknown.
# Needs sdcguard on the node (or in the NPD image) and access to the GPUs.
exec sdcguard nhc --minutes "${SDCGUARD_MINUTES:-2}" --out "${SDCGUARD_OUT:-/var/lib/sdcguard/nhc}"
