#!/bin/bash
# Screen one node's GPUs with sdcguard and leave the result where the health check reads it.
# Run it on idle nodes, for example in a low-priority maintenance partition, once a day per node:
#   sbatch --nodelist=gpu-node-17 deploy/slurm/sdcguard-screen.sbatch
# The output folder must be writable by the user the job runs as (create it once with
#   install -d -o <that user> /var/lib/sdcguard/nhc   on every node). Set --gres to your nodes' GPU count.
#SBATCH --job-name=sdcguard-screen
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/sdcguard-screen-%j.log
sdcguard nhc --minutes "${SDCGUARD_MINUTES:-10}" --out "${SDCGUARD_OUT:-/var/lib/sdcguard/nhc}"
