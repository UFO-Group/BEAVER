#!/bin/bash
#SBATCH -J xgboost_mlp_svm
#SBATCH -p gpu
#SBATCH -w gpu4
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/task/xgboost_mlp_svm-%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/task/xgboost_mlp_svm-%J.out

export CUDA_VISIBLE_DEVICES=1
cd "$SLURM_SUBMIT_DIR"

PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/baseline_xgboost_mlp_svm.py"

NAME="MeltingTemperature"
TARGET="Melting Temperature (°C)"
IN="/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/MeltingTemperature-pooled-4096.csv"
OUTROOT="/home/fangxr/ML_Tool/2.ML/Result/MeltingTemperature"

# "MeltingTemperature"
# "GlassTransitionTemperature"
# "ElongationBreak"
# "YoungsModulus"
# "TensileStrength"
  
# "Melting Temperature (°C)"
# "Glass Transition Temperature (°C)"
# "Elongation at Break (%) log10"
# "Young's Modulus (kPa) log10"
# "Tensile Strength (MPa) log10"

OUT_XGB="${OUTROOT}/xgb"
OUT_MLP="${OUTROOT}/mlp"
OUT_SVM="${OUTROOT}/svm"

mkdir -p "$OUT_XGB" "$OUT_MLP" "$OUT_SVM"

echo "[$(date)] Running XGBoost..."
$PY "$SCRIPT" \
  --in_csv "$IN" \
  --target "$TARGET" \
  --model xgb \
  --save_dir "$OUT_XGB" \
  --cv10 --cv_folds 10 --save_train_pred \
  --seed 42 \
  --xgb_n_estimators 1000 \
  --xgb_learning_rate 0.01 \
  --xgb_max_depth 8 \
  --xgb_subsample 0.7 \
  --xgb_colsample_bytree 0.3 \
  --xgb_reg_lambda 3.0

echo "[$(date)] Running MLP..."
$PY "$SCRIPT" \
  --in_csv "$IN" \
  --target "$TARGET" \
  --model mlp \
  --save_dir "$OUT_MLP" \
  --cv10 --cv_folds 10 --save_train_pred \
  --mlp_hidden "512,256,128" \
  --mlp_activation tanh \
  --mlp_alpha 1e-3 \
  --mlp_lr 1e-3 \
  --mlp_max_iter 1000 \
  --mlp_early_stop

echo "[$(date)] Running SVM..."
$PY "$SCRIPT" \
  --in_csv "$IN" \
  --target "$TARGET" \
  --model svm \
  --save_dir "$OUT_SVM" \
  --cv10 --cv_folds 10 --save_train_pred \
  --svm_kernel rbf \
  --svm_C 50.0 \
  --svm_epsilon 0.2 \
  --svm_gamma auto \
  --no_perm

echo "Done MeltingTemperature."
