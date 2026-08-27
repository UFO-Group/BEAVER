#!/bin/bash
#SBATCH -J predict_xgb
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/predict/task/%J.out
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/predict/task/%J.err

cd "$SLURM_SUBMIT_DIR"

# ===== 按需修改 =====
PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/predict/predict_xgb.py"

IN_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB_morgan.csv"
SOURCE_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB.csv"
OUT_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/Result/Prediction-MeltingTemperature-xgb.csv"
MODEL_DIR="/home/fangxr/ML_Tool/2.ML/Result/MeltingTemperature/xgb/fold_models/"
TARGET_NAME="Melting Temperature (°C)"

mkdir -p /home/fangxr/ML_Tool/2.ML/predict/task
mkdir -p "$(dirname "$OUT_CSV")"

$PY "$SCRIPT" \
  --in_csv "$IN_CSV" \
  --source_csv "$SOURCE_CSV" \
  --out_csv "$OUT_CSV" \
  --model_dir "$MODEL_DIR" \
  --model_name "best_model.joblib" \
  --target_name "$TARGET_NAME"
