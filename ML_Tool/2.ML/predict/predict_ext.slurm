#!/bin/bash
#SBATCH -J predict_ext
#SBATCH -p gpu
#SBATCH -w gpu4
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/predict/task/%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/predict/task/%J.out

cd "$SLURM_SUBMIT_DIR"

PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/predict/predict_ext.py"

# ===== 按需修改 =====
IN_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB_morgan.csv"
SOURCE_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB.csv"
OUT_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/Result/Prediction-ElongationBreak-EXT.csv"
MODEL_DIR="/home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/extratrees/extratrees/fold_models/"
TARGET_NAME="Elongation at Break (%) log10"
ID_COL="row_index"

#"/home/fangxr/ML_Tool/2.ML/Result/ElongationBreak/extratrees/extratrees/predictions/"
#"/home/fangxr/ML_Tool/2.ML/Result/YoungsModulus/extratrees/extratrees/fold_models/"

mkdir -p /home/fangxr/ML_Tool/2.ML/predict/task
mkdir -p "$(dirname "$OUT_CSV")"

echo "[INFO] 开始 ExtraTrees 预测"
echo "[INFO] IN_CSV=$IN_CSV"
echo "[INFO] SOURCE_CSV=$SOURCE_CSV"
echo "[INFO] OUT_CSV=$OUT_CSV"
echo "[INFO] MODEL_DIR=$MODEL_DIR"
echo "[INFO] TARGET_NAME=$TARGET_NAME"

$PY "$SCRIPT" \
  --in_csv "$IN_CSV" \
  --source_csv "$SOURCE_CSV" \
  --out_csv "$OUT_CSV" \
  --model_dir "$MODEL_DIR" \
  --target_name "$TARGET_NAME" \
  --id_col "$ID_COL"

RET=$?
echo "[INFO] predict_ext.py exit code = $RET"
exit $RET
