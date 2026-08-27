#!/bin/bash
#SBATCH -J baseline
#SBATCH -p gpu
#SBATCH -w gpu3
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/draw/task/%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/draw/task/%J.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

cd "$SLURM_SUBMIT_DIR"

python /home/fangxr/ML_Tool/2.ML/draw/draw_r2.py \
  --train_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/rf_cv10/fold_08_train.csv \
  --test_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/rf_cv10/fold_08_valid.csv \
  --outdir /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/rf_cv10/draw

# python /home/fangxr/ML_Tool/2.ML/draw/draw_r2.py \
#   --train_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/mlp/fold_08_train.csv \
#   --test_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/mlp/fold_08_valid.csv \
#   --outdir /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/mlp/draw
#   
# python /home/fangxr/ML_Tool/2.ML/draw/draw_r2.py \
#   --train_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/svm/fold_05_train.csv \
#   --test_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/svm/fold_05_valid.csv \
#   --outdir /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/svm/draw
# 
# python /home/fangxr/ML_Tool/2.ML/draw/draw_r2.py \
#   --train_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/rf_cv10/fold_08_train.csv \
#   --test_csv /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/rf_cv10/fold_08_valid.csv \
#   --outdir /home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/rf_cv10/draw

  
  
  
