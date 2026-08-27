#!/bin/bash
#SBATCH -J knn_gpr_MT
#SBATCH -p gpu
#SBATCH -w gpu3
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/task/%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/task/%J.out

export CUDA_VISIBLE_DEVICES=2
cd "$SLURM_SUBMIT_DIR"

PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/KNN_GP.py"

NAME="TensileStrength"
TARGET="Tensile Strength (MPa) log10"
IN="/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/TensileStrength-pooled-4096.csv"
OUTROOT="/home/fangxr/ML_Tool/2.ML/Result"
OUT_knn="${OUTROOT}/${NAME}/knn"
OUT_gpr="${OUTROOT}/${NAME}/gpr"

# "Melting Temperature (°C)"
# "Glass Transition Temperature (°C)"
# "Elongation at Break (%) log10"
# "Young's Modulus (kPa) log10"
# "Tensile Strength (MPa) log10"

# "MeltingTemperature")
# "GlassTransitionTemperature")
# "ElongationBreak")
# "YoungsModulus")
# "TensileStrength")

mkdir -p "$OUT_knn" "$OUT_gpr"

echo "====== Running ${NAME} ======"

$PY "$SCRIPT" \
    --in_csv "$IN" \
    --target_col "$TARGET" \
    --models knn \
    --cv_folds 10 \
    --knn_n_neighbors 9 \
    --n_jobs 8 \
    --random_state 42 \
    --save_model \
    --out_dir "$OUT_knn"

$PY "$SCRIPT" \
    --in_csv "$IN" \
    --target_col "$TARGET" \
    --models gpr \
    --cv_folds 10 \
    --random_state 42 \
    --save_model \
    --out_dir "$OUT_gpr"

echo "[Finished] ${NAME}"
