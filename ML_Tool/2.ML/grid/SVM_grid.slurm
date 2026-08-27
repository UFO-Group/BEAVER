#!/bin/bash
#SBATCH -J svm_melt
#SBATCH -p gpu
#SBATCH -w gpu2
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/grid/task/svm_melt-%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/grid/task/svm_melt-%J.out

export CUDA_VISIBLE_DEVICES=0
cd "$SLURM_SUBMIT_DIR"

PYTHON=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/grid/grid_svm.py"

TARGET="Tensile Strength (MPa) log10"
INPUT="/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/TensileStrength-pooled-4096.csv"
OUTDIR="/home/fangxr/ML_Tool/2.ML/grid/outputs/TensileStrength/svm_grid_loop-TensileStrength"

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

echo "Running SVM Grid Search: MeltingTemperature"
$PYTHON "$SCRIPT" \
  --in_csv "$INPUT" \
  --target "$TARGET" \
  --save_dir "$OUTDIR"

echo "[DONE] MeltingTemperature"
