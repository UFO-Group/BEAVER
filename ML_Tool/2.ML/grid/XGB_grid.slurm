#!/bin/bash
#SBATCH -J xgb_melt
#SBATCH -p gpu
#SBATCH -w gpu2
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/grid/task/xgb_melt-%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/grid/task/xgb_melt-%J.out

export CUDA_VISIBLE_DEVICES=0
cd "$SLURM_SUBMIT_DIR"

PYTHON=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/grid/grid_xgb.py"

TARGET="Melting Temperature (°C)"
INPUT="/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/MeltingTemperature-pooled-4096.csv"
OUTDIR="/home/fangxr/ML_Tool/2.ML/grid/outputs/MeltingTemperature/xgb_grid_loop-MeltingTemperature"

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


mkdir -p "$OUTDIR"

echo "Running XGB Grid Search: YoungsModulus"
$PYTHON "$SCRIPT" \
    --in_csv "$INPUT" \
    --target "$TARGET" \
    --save_dir "$OUTDIR"

echo "[$(date)] Finished YoungsModulus"
