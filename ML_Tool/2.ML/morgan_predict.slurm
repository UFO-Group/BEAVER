#!/bin/bash
#SBATCH -J baseline
#SBATCH -p gpu
#SBATCH -w gpu3
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/task/%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/task/%J.out

export CUDA_VISIBLE_DEVICES=0

cd "$SLURM_SUBMIT_DIR"

PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/morgan_pooling_new.py"

NAMES=(
  "SMILEABC"
)

IN_CSVS=(
  "/home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEABC.csv"
)

OUT_CSVS=(
  "/home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEABC_morgan.csv"
)

# 明确我们要保留的目标列数组（空格分隔）
TARGET_COLS=("Polymer A" "Polymer B" "Polymer C")

POLYMER_COLS=("SMILE A" "SMILE B" "SMILE C")
RADIUS=3
NBITS=2048
FEATURE_MODE=2048
DR_METHOD="none"

NUM_TASKS=${#NAMES[@]}

for ((i=0; i<NUM_TASKS; i++));
do
  NAME="${NAMES[$i]}"
  IN_CSV="${IN_CSVS[$i]}"
  OUT_CSV="${OUT_CSVS[$i]}"

  echo "=============================="
  echo "[TASK $((i+1))/${NUM_TASKS}] ${NAME}"
  echo "  IN  : ${IN_CSV}"
  echo "  OUT : ${OUT_CSV}"
  echo "  Y   : ${TARGET_COLS[*]}"  # 这里打印出数组的所有元素，方便检查
  echo "=============================="

  mkdir -p "$(dirname "$OUT_CSV")"

  # 关键点：用 "${TARGET_COLS[@]}" 将两个列名作为独立参数传给 python
  $PY "$SCRIPT" \
    --in_csv "$IN_CSV" \
    --out_csv "$OUT_CSV" \
    --polymer_cols "${POLYMER_COLS[@]}" \
    --radius "$RADIUS" \
    --nbits "$NBITS" \
    --feature_mode "$FEATURE_MODE" \
    --target_cols "${TARGET_COLS[@]}" \
    --dr_method "$DR_METHOD"

  echo "[DONE] pooled CSV: $OUT_CSV"
done

echo "[ALL DONE] Morgan pooling finished for ${NUM_TASKS} tasks."
