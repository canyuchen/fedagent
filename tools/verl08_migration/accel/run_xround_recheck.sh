#!/bin/bash
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh 2>/dev/null
conda activate fedagent-verl08
cd "$REPO"
rm -rf _scratch/accel/xround_recheck_out
sed 's#xround_full_out#xround_recheck_out#' _scratch/accel/xround_full.yaml > _scratch/accel/xround_recheck.yaml
echo "XROUND RECHECK START $(date +%s)"
python -u -m fedagent.fed.run_fed --config _scratch/accel/xround_recheck.yaml
echo "XROUND RECHECK rc=$? end=$(date +%s)"
grep -c "Started a local Ray instance" _scratch/accel/xround_recheck_out/round_*/persistent_training.log 2>/dev/null | head -1
ls -d _scratch/accel/xround_recheck_out/round_*/aggregated/checkpoints/global_step_0/actor 2>/dev/null
