#!/bin/bash
#SBATCH -J rf_grid1
#SBATCH -p gpu
#SBATCH -w gpu3
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/grid/task/rf_melt-%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/grid/task/rf_melt-%J.out

export CUDA_VISIBLE_DEVICES=1

cd "$SLURM_SUBMIT_DIR"

PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/grid/rf_grid_loop.py"
# SCRIPT="/home/fangxr/ML_Tool/2.ML/grid/rf_grid_loop.py"

TEST_SIZE=0.2

NAME="GlassTransitionTemperature"
TARGET="Glass Transition Temperature (°C)"
CSV="/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/GlassTransitionTemperature-pooled-4096.csv"
OUTDIR="/home/fangxr/ML_Tool/2.ML/grid/outputs/GlassTransitionTemperature/rf_grid–loop-GlassTransitionTemperature"

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

echo "=============================="
echo "[TASK] ${NAME}"
echo "  CSV   = ${CSV}"
echo "  TARGET= ${TARGET}"
echo "  OUT   = ${OUTDIR}"
echo "=============================="

mkdir -p "$OUTDIR"

$PY "$SCRIPT" \
  --in_csv "$CSV" \
  --target "$TARGET" \
  --save_dir "$OUTDIR" \
  --id_cols SampleID,RecipeID \
  --test_size "$TEST_SIZE"

echo "[DONE] ${NAME} -> ${OUTDIR}"
