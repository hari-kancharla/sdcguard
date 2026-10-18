#!/bin/bash
# Slurm HealthCheckProgram (slurm.conf: HealthCheckProgram=/opt/sdcguard/healthcheck.sh).
# Slurm kills a HealthCheckProgram that runs longer than 60 seconds, so this doesn't screen anything: it
# reads the result of the last screen (sdcguard-screen.sbatch, run on its own schedule; reading takes well
# under a second) and drains the node if a GPU computed wrong results. Results older than 36 hours count
# as unknown (logged, no drain). slurmd's PATH may not include sdcguard: set SDCGUARD_BIN to its full path.
BIN="${SDCGUARD_BIN:-sdcguard}"
OUT="${SDCGUARD_OUT:-/var/lib/sdcguard/nhc}"
NODE="${SLURMD_NODENAME:-$(hostname -s)}"
MSG=$(timeout 30 "$BIN" nhc --cached --out "$OUT" 2>/dev/null)
RC=$?
if [ "$RC" = "1" ]; then
    scontrol update NodeName="$NODE" State=DRAIN Reason="sdcguard: ${MSG:0:200}"
elif [ "$RC" != "0" ]; then
    logger -t sdcguard "health check: ${MSG:-no result (exit $RC)}"
fi
exit 0
