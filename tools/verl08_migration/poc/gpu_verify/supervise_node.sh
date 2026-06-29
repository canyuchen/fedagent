#!/bin/bash
# args: JOBID  CUR_OUT  [DRIVER OUT]...
JOBID=$1; shift; CUR=$1; shift
echo "[supervise $JOBID] waiting for current ($CUR) to finish @ $(date)"
for i in $(seq 1 260); do grep -qE "exit=" "$CUR" 2>/dev/null && break; sleep 60; done
while [ $# -ge 2 ]; do
  D=$1; O=$2; shift 2
  echo "[supervise $JOBID] launching $D -> $O @ $(date)"
  setsid nohup srun --overlap --jobid="$JOBID" -N1 bash "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/_scratch/gpu_verify/$D" < /dev/null > "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/_scratch/gpu_verify/$O" 2>&1 &
  sleep 45
  for i in $(seq 1 260); do grep -qE "exit=" "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/_scratch/gpu_verify/$O" 2>/dev/null && break; sleep 60; done
  echo "[supervise $JOBID] $D done @ $(date)"
done
echo "[supervise $JOBID] QUEUE EXHAUSTED @ $(date) -- refill needed"
