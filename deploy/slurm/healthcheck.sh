#!/bin/bash
# Slurm HealthCheckProgram (slurm.conf: HealthCheckProgram=/opt/sdcguard/healthcheck.sh).
# Slurm kills a HealthCheckProgram that runs longer than 60 seconds, so this doesn't screen anything: it
# reads the result of the last screen (sdcguard-screen.sbatch, run on its own schedule; reading takes well
# under a second) and drains the node if a GPU computed wrong results, or resumes a node it drained once a newer
# screen passes; a node drained for another reason is left as it is. Results older than 36 hours count as
# unknown (logged, no drain). slurmd's PATH may not include sdcguard: set SDCGUARD_BIN to its full path.
BIN="${SDCGUARD_BIN:-sdcguard}"
OUT="${SDCGUARD_OUT:-/var/lib/sdcguard/nhc}"
NODE="${SLURMD_NODENAME:-$(hostname -s)}"
MSG=$(timeout 30 "$BIN" nhc --cached --out "$OUT" 2>/dev/null)
RC=$?
INFO=$(scontrol -o show node "$NODE" 2>/dev/null)
STATE=$(printf '%s\n' "$INFO" | tr ' ' '\n' | sed -n 's/^State=//p' | head -n 1)
OURS=0
case "$INFO" in *" Reason=sdcguard:"*) OURS=1 ;; esac
if [ "$RC" = "1" ]; then
    case "$STATE" in
        *DRAIN*|*DOWN*|*FAIL*)
            if [ "$OURS" = "0" ]; then
                # drained already for another reason (maintenance, another check): that reason stays, so this
                # script never later resumes a node someone else took out; once they resume it, the next run
                # drains it as its own
                logger -t sdcguard "health check: ${MSG:0:200} (node already drained for another reason; left as is)"
            else
                scontrol update NodeName="$NODE" State=DRAIN Reason="sdcguard: ${MSG:0:200}"
            fi ;;
        *) scontrol update NodeName="$NODE" State=DRAIN Reason="sdcguard: ${MSG:0:200}" ;;
    esac
elif [ "$RC" = "0" ]; then
    # A newer screen passed. Return the node to service only if this script drained it (its reason starts with
    # "sdcguard:"; LBNL NHC likewise onlines only nodes whose note it set, marked "NHC:"): any other drain stays.
    if [ "$OURS" = "1" ]; then
        scontrol update NodeName="$NODE" State=RESUME
    fi
else
    logger -t sdcguard "health check: ${MSG:-no result (exit $RC)}"
fi
exit 0
