#!/bin/bash
#SBATCH -J predict_rf_one
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/predict/task/%x_%j.out
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/predict/task/%x_%j.err

set -euo pipefail

PY=python
MODEL_DIR="/home/fangxr/ML_Tool/2.ML/Result/GlassTransitionTemperature/rf_cv10/"

IN_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB_morgan.csv"
OUT_CSV="/home/fangxr/ML_Tool/2.ML/predict/outputs/Result/Prediction-GlassTransitionTemperature-rf.csv"

RADIUS=3
NBITS=2048
FP_TYPE="count"
POOLING="avg"
SMILES_COLS="SMILE A,SMILE B"
TARGET_NAME="Glass Transition Temperature (°C)"

echo "[DONE] 写出：$OUT_CSV"

python /home/fangxr/ML_Tool/2.ML/predict/predict_rf.py \
  --in_csv /home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB_morgan.csv \
  --source_csv /home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB.csv \
  --out_csv /home/fangxr/ML_Tool/2.ML/predict/outputs/Result/Prediction-GlassTransitionTemperature-rf.csv \
  --model_dir /home/fangxr/ML_Tool/2.ML/Result/GlassTransitionTemperature/rf_cv10/ \
  --target_name "Glass Transition Temperature (°C)"
  
python /home/fangxr/ML_Tool/2.ML/predict/predict_rf.py \
  --in_csv /home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB_morgan.csv \
  --source_csv /home/fangxr/ML_Tool/2.ML/predict/outputs/SMILEAB.csv \
  --out_csv /home/fangxr/ML_Tool/2.ML/predict/outputs/Result/Prediction-TensileStrength-rf.csv \
  --model_dir /home/fangxr/ML_Tool/2.ML/Result/TensileStrength/rf_cv10/ \
  --target_name "Tensile Strength (MPa) log10"
  
