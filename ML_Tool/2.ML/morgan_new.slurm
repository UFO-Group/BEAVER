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

# ===== 5 个任务定义 =====
NAMES=(
  "TensileStrength"
  "ElongationBreak"
  "YoungsModulus"
  #"GlassTransitionTemperature"
  #"MeltingTemperature"
)

IN_CSVS=(
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/TensileStrength-log10.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/ElongationBreak-log10.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/YoungsModulus-log10.csv"
  #"/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/GlassTransitionTemperature-linear.csv"
  #"/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/MeltingTemperature-linear.csv"
)

OUT_CSVS=(
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/TensileStrength-pooled-4096.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/ElongationBreak-pooled-4096.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/YoungsModulus-pooled-4096.csv"
  #"/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/GlassTransitionTemperature-pooled-4096.csv"
  #"/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/MeltingTemperature-pooled-4096.csv"
)

TARGETS=(
  "Tensile Strength (MPa) log10"
  "Elongation at Break (%) log10"
  "Young's Modulus (kPa) log10"
  #"Glass Transition Temperature (°C)"
  #"Melting Temperature (°C)"
)

POLYMER_COLS=( "SMILE A" "SMILE B" "SMILE C" "SMILE D" "SMILE E" )
RADIUS=3
NBITS=2048
FEATURE_MODE=2048
DR_METHOD="none"

NUM_TASKS=${#NAMES[@]}

for ((i=0; i<NUM_TASKS; i++)); do
  NAME="${NAMES[$i]}"
  IN_CSV="${IN_CSVS[$i]}"
  OUT_CSV="${OUT_CSVS[$i]}"
  TARGET_COL="${TARGETS[$i]}"

  echo "=============================="
  echo "[TASK $((i+1))/${NUM_TASKS}] ${NAME}"
  echo "  IN  : ${IN_CSV}"
  echo "  OUT : ${OUT_CSV}"
  echo "  Y   : ${TARGET_COL}"
  echo "=============================="

  mkdir -p "$(dirname "$OUT_CSV")"

  $PY "$SCRIPT" \
    --in_csv "$IN_CSV" \
    --out_csv "$OUT_CSV" \
    --polymer_cols "${POLYMER_COLS[@]}" \
    --radius "$RADIUS" \
    --nbits "$NBITS" \
    --feature_mode "$FEATURE_MODE" \
    --target_cols "$TARGET_COL" \
    --dr_method "$DR_METHOD"
#   --feature_mode 2055 \
#   --feature_mode extended \
#   --dr_method pca \
#   --dr_dim 256
#   --dr_method umap \
#   --dr_dim 128
  echo "[DONE] pooled CSV: $OUT_CSV"
  echo "[DONE] pooled Parquet: ${OUT_CSV%.csv}.parquet"
done

echo "[ALL DONE] Morgan pooling finished for ${NUM_TASKS} targets."


