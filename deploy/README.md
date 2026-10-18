# Running sdcguard across a fleet

`sdcguard nhc` is a node health check: it screens every GPU on the node with a real transformer workload
(every matmul checked), prints one line and exits `0` (all GPUs computed correctly), `1` (a GPU computed
wrong results; evidence saved) or `2` (couldn't tell). Those are the exit codes Kubernetes
node-problem-detector custom plugins use, and what Slurm's NHC needs to drain a node.

| File | What it does |
|---|---|
| `Dockerfile` | an image with sdcguard (PyTorch 2.14.1, CUDA 13.0); its default command is `sdcguard nhc --minutes 2` |
| `kubernetes/node-problem-detector/sdcguard-monitor.json` | node-problem-detector custom plugin: sets the node condition `GPUComputeIncorrect` when a GPU fails |
| `kubernetes/node-problem-detector/sdcguard-check.sh` | the script that plugin runs |
| `kubernetes/screen-job.yaml` | a one-off screen of one node (e.g. when a node joins, or in a maintenance window); taints the node if a GPU fails |
| `slurm/sdcguard-screen.sbatch` | screens one node's GPUs (run it on idle nodes on a schedule); leaves the result for the health checks below |
| `slurm/sdcguard.nhc` | a check for LBNL NHC (`check_sdcguard`): reads the last screen's result instantly; NHC drains the node when a GPU failed |
| `slurm/healthcheck.sh` | the same for Slurm's plain `HealthCheckProgram`, draining the node itself |

Slurm kills a `HealthCheckProgram` that runs longer than 60 seconds, so on Slurm the screen and the health
check are separate: the batch job screens (10 minutes by default) and writes `status.json`; the health checks
run `sdcguard nhc --cached`, which only reads it. A failure keeps the node drained until a new screen passes;
a result older than 36 hours counts as unknown (logged, not drained).

Screening needs the GPUs: run it when a node joins, between jobs, or in maintenance windows (as GPU clouds
already do with their own idle-node tests), not alongside a training job. Inside running jobs, use the
vLLM plugin (`SDCGUARD=1`) or the training check instead; they check the job's own math.

After a screen, `sdcguard report <out dirs> --out fleet_report` turns the results of many nodes into one
page, a quarantine list and an evidence zip per failed GPU for the RMA.
