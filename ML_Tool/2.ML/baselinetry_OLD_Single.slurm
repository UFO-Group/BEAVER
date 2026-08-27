#!/bin/bash
#SBATCH -J MT_rf
#SBATCH -p gpu
#SBATCH -w gpu4
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/task/%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/task/%J.out

export CUDA_VISIBLE_DEVICES=0

PY=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/baseline_OLD.py"  # 你的训练脚本
GROUP_COL="RecipeID"
SEED=42
USE_YLOG=0               # 1=对y取log1p；0=不取

# ---- 记录运行条件的小函数 ----
tag_run () {
  local DIR="$1"; local IN_CSV="$2"; local TARGET="$3"; local MODEL="$4"; local YLOG="$5"
  local CV_ON="$6"; local CV_KIND="$7"; local GROUP="$8"; local SEED_VAL="$9"; local OOF_PATH="${10:-}"
  mkdir -p "$DIR"
  cat > "$DIR/run_conditions.json" <<JSON
{"timestamp":"$(date -Iseconds)",
 "in_csv":"${IN_CSV}",
 "target":"${TARGET}",
 "script":"${SCRIPT}",
 "model":"${MODEL}",
 "y_log1p":${YLOG},
 "cv10":${CV_ON},
 "cv_kind":"${CV_KIND}",
 "group_col":"${GROUP}",
 "seed":${SEED_VAL},
 "oof_path":"${OOF_PATH}"}
JSON
  echo "[COND] model=${MODEL} target=${TARGET} y_log1p=${YLOG} cv=${CV_ON} kind=${CV_KIND} group_col=${GROUP:-None} seed=${SEED_VAL} in_csv=$(basename "$IN_CSV") save_dir=${DIR}"
}

# ========= 当前任务配置 =========
NAME="GlassTransitionTemperature"
TARGET="Glass Transition Temperature (°C)"
TRAIN_CSV="/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/GlassTransitionTemperature-pooled-4096.csv"
OUTROOT="/home/fangxr/ML_Tool/2.ML/Result/GlassTransitionTemperature"
# "Melting Temperature (°C)"
# "Glass Transition Temperature (°C)"
# "Elongation at Break (%) log10"
# "Young's Modulus (kPa) log10"
# "Tensile Strength (MPa) log10"

echo "=============================="
echo "[TASK] ${NAME}"
echo "  CSV   = ${TRAIN_CSV}"
echo "  TARGET= ${TARGET}"
echo "  OUT   = ${OUTROOT}"
echo "=============================="

# ---- 针对当前任务选择 RF 超参数 ----
RF_ARGS=( --rf_n_estimators 300 --rf_max_depth 20 --rf_min_samples_split 8 --rf_min_samples_leaf 1 --rf_max_features 0.1 )


# "MeltingTemperature")
#用     RF_ARGS=( --rf_n_estimators 200 --rf_max_depth 15 --rf_min_samples_split 10 --rf_min_samples_leaf 4 --rf_max_features 0.5)
#更新     RF_ARGS=( --rf_n_estimators 200 --rf_max_depth 15 --rf_min_samples_split 14 --rf_min_samples_leaf 2 --rf_max_features 0.5)

# "GlassTransitionTemperature")
#	  RF_ARGS=( --rf_n_estimators 600 --rf_max_depth 25 --rf_min_samples_split 10 --rf_min_samples_leaf 1 --rf_max_features 0.2 )
#	  RF_ARGS=( --rf_n_estimators 300 --rf_max_depth 20 --rf_min_samples_split 8 --rf_min_samples_leaf 1 --rf_max_features 0.1 )

# "ElongationBreak")
#     RF_ARGS=( --rf_n_estimators 300 --rf_max_depth 30 --rf_min_samples_split 8 --rf_min_samples_leaf 2 --rf_max_features 0.1 )
#     RF_ARGS=( --rf_n_estimators 200 --rf_max_depth 35 --rf_min_samples_split 8 --rf_min_samples_leaf 2 --rf_max_features 0.1 )

# "YoungsModulus")
#     RF_ARGS=( --rf_n_estimators 300 --rf_max_depth 25 --rf_min_samples_split 10 --rf_min_samples_leaf 1 --rf_max_features 0.2 )
#     RF_ARGS=( --rf_n_estimators 1000 --rf_max_depth 35 --rf_min_samples_split 8 --rf_min_samples_leaf 1 --rf_max_features 0.1 )

# "TensileStrength")
#     RF_ARGS=( --rf_n_estimators 100 --rf_max_depth 20 --rf_min_samples_split 12 --rf_min_samples_leaf 2 --rf_max_features 0.3 )
#     RF_ARGS=( --rf_n_estimators 400 --rf_max_depth 25 --rf_min_samples_split 8 --rf_min_samples_leaf 4 --rf_max_features 0.5 )


echo "[INFO] RF_ARGS for ${NAME}: ${RF_ARGS[*]}"

# ---- 检查分组列是否可用（对当前 CSV）----
HAS_GROUP=$($PY - <<PY
import pandas as pd
csv=r"$TRAIN_CSV"; col=r"$GROUP_COL"
try:
    df = pd.read_csv(csv, usecols=[col], dtype='string', low_memory=False)
    print(1 if df[col].nunique(dropna=True) >= 2 else 0)
except Exception:
    print(0)
PY
)
echo "[INFO] group_col usable for ${NAME}? $HAS_GROUP"

# ---- 公共参数 ----
COMMON=( --in_csv "$TRAIN_CSV" --target "$TARGET" --seed "$SEED" --id_cols SampleID,RecipeID --save_train_pred )
if [ "$USE_YLOG" -eq 1 ]; then COMMON+=( --y_log1p ); fi

# ===== RF · 10-fold CV =====
if [ "$HAS_GROUP" -eq 1 ]; then
  SAVE_DIR_CV10="${OUTROOT}/rf_cv10_group"
  mkdir -p "$SAVE_DIR_CV10"
  $PY "$SCRIPT" \
    "${COMMON[@]}" \
    --model rf \
    --cv10 --cv_folds 10 \
    --group_col "$GROUP_COL" \
    --save_dir "$SAVE_DIR_CV10" \
    --oof_path "$SAVE_DIR_CV10/cv10_oof.csv" \
    "${RF_ARGS[@]}"
  tag_run "$SAVE_DIR_CV10" "$TRAIN_CSV" "$TARGET" "rf" "$USE_YLOG" 1 "GroupKFold(10 or ≤n_groups)" "$GROUP_COL" "$SEED" "$SAVE_DIR_CV10/cv10_oof.csv"
else
  SAVE_DIR_CV10="${OUTROOT}/rf_cv10"
  mkdir -p "$SAVE_DIR_CV10"
  $PY "$SCRIPT" \
    "${COMMON[@]}" \
    --model rf \
    --cv10 --cv_folds 10 \
    --save_dir "$SAVE_DIR_CV10" \
    --oof_path "$SAVE_DIR_CV10/cv10_oof.csv" \
    "${RF_ARGS[@]}"
  tag_run "$SAVE_DIR_CV10" "$TRAIN_CSV" "$TARGET" "rf" "$USE_YLOG" 1 "KFold(10, shuffle=True)" "" "$SEED" "$SAVE_DIR_CV10/cv10_oof.csv"
fi

echo "[DONE] ${NAME} -> ${OUTROOT}"
