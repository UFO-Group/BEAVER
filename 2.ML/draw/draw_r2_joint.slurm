#!/bin/bash
#SBATCH -J draw_r2_joint
#SBATCH -p gpu
#SBATCH -w gpu3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/draw/task/%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/draw/task/%J.out

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

cd "$SLURM_SUBMIT_DIR"
mkdir -p /home/fangxr/ML_Tool/2.ML/draw/task

python /home/fangxr/ML_Tool/2.ML/draw/draw_r2_joint.py \
  --train_csv /home/fangxr/ML_Tool/2.ML/Result/TensileStrength/rf_cv10/fold_05_train.csv \
  --test_csv /home/fangxr/ML_Tool/2.ML/Result/TensileStrength/rf_cv10/fold_05_valid.csv \
  --outdir /home/fangxr/ML_Tool/2.ML/Result/TensileStrength/rf_cv10/draw_kde_attached/ \
  --linear_tick_nbins 6 \
  --log_major_step 0.5 \
  --bandwidth_scale 1.15 \
  --marginal_size 20%



# /home/fangxr/ML_Tool/2.ML/Result/MeltingTemperature/xgb/fold_07_train.csv ???fold_09_valid.csv
# /home/fangxr/ML_Tool/2.ML/Result/GlassTransitionTemperature/rf_cv10/fold_09_train.csv ???fold_09_valid.csv
# /home/fangxr/ML_Tool/2.ML/Result/YoungsModulus/extratrees/extratrees/predictions/fold_8_train_predictions.csv ???fold_8_test_predictions.csv
# /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/extratrees/extratrees/predictions/fold_9_train_predictions.csv ???fold_9_test_predictions.csv
# /home/fangxr/ML_Tool/2.ML/Result/TensileStrength/rf_cv10/fold_05_train.csv ???fold_05_valid.csv
