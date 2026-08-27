#!/bin/bash
#SBATCH -J knn_grid
#SBATCH -p gpu
#SBATCH -w gpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-4
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/grid/task/%A_%a.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/grid/task/%A_%a.out

export CUDA_VISIBLE_DEVICES=3
cd "$SLURM_SUBMIT_DIR"

mkdir -p /home/fangxr/ML_Tool/2.ML/grid/task

PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/grid/knn_grid_search.py"

IDX=${SLURM_ARRAY_TASK_ID}

NAMES=(
  "MeltingTemperature"
  "GlassTransitionTemperature"
  "ElongationBreak"
  "YoungsModulus"
  "TensileStrength"
)

TARGETS=(
  "Melting Temperature (°C)"
  "Glass Transition Temperature (°C)"
  "Elongation at Break (%) log10"
  "Young's Modulus (kPa) log10"
  "Tensile Strength (MPa) log10"
)

INPUTS=(
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/MeltingTemperature-pooled-4096.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/GlassTransitionTemperature-pooled-4096.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/ElongationBreak-pooled-4096.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/YoungsModulus-pooled-4096.csv"
  "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/TensileStrength-pooled-4096.csv"
)

NAME="${NAMES[$IDX]}"
TARGET="${TARGETS[$IDX]}"
IN="${INPUTS[$IDX]}"
OUTROOT="/home/fangxr/ML_Tool/2.ML/grid/outputs"
OUTDIR="${OUTROOT}/${NAME}/knn"

mkdir -p "$OUTDIR"

N_JOBS="${SLURM_CPUS_PER_TASK:-6}"

echo "====== Running KNN Grid Search: ${NAME} ======"
echo "IDX      = $IDX"
echo "INPUT    = $IN"
echo "TARGET   = $TARGET"
echo "OUTDIR   = $OUTDIR"
echo "N_JOBS   = $N_JOBS"

$PY "$SCRIPT" \
    --in_csv "$IN" \
    --target_col "$TARGET" \
    --cv_folds 10 \
    --random_state 42 \
    --n_jobs "$N_JOBS" \
    --refit_metric neg_rmse \
    --save_model \
    --out_dir "$OUTDIR"

echo "[Finished] KNN Grid Search - ${NAME}"
