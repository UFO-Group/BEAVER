#!/bin/bash
#SBATCH -J baseline
#SBATCH -p gpu
#SBATCH -w gpu4
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/task/%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/task/%J.out

export CUDA_VISIBLE_DEVICES=0
cd "$SLURM_SUBMIT_DIR"

PYTHON=python
SCRIPT="/home/fangxr/ML_Tool/2.ML/LGBM_Catboost_Extratree.py"

# ===== 5 个任务定义 =====
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

OUTROOT="/home/fangxr/ML_Tool/2.ML/Result/"

# ===== 通用参数 =====
COMMON_ARGS=(
  --cv_folds 10
  --test_size 0.1
  --random_state 42
  --save_model
)

NUM_TASKS=${#NAMES[@]}

for ((i=0; i<NUM_TASKS; i++)); do
  NAME="${NAMES[$i]}"
  TARGET="${TARGETS[$i]}"
  INPUT="${INPUTS[$i]}"

  OUT_CAT="${OUTROOT}/${NAME}/catboost"
  OUT_LGBM="${OUTROOT}/${NAME}/lgbm"
  OUT_ETR="${OUTROOT}/${NAME}/extratrees"

  mkdir -p "$OUT_CAT" "$OUT_LGBM" "$OUT_ETR"

  echo "===================================================="
  echo "[TASK $((i+1))/${NUM_TASKS}] ${NAME}"
  echo " CSV    : ${INPUT}"
  echo " TARGET : ${TARGET}"
  echo "===================================================="

  # =========================================================
  # 每个数据集单独设置 3 个模型的超参数
  # 下面这些数值请替换成你网格搜索/调参得到的最佳参数
  # =========================================================
  case "$NAME" in
    "MeltingTemperature")
      CAT_ARGS=(
        --cat_iterations 600
        --cat_learning_rate 0.05
        --cat_depth 8
        --cat_l2_leaf_reg 3
        --cat_subsample 0.8
        --n_jobs 6
      )
      LGBM_ARGS=(
        --lgbm_n_estimators 400
        --lgbm_learning_rate 0.03
        --lgbm_num_leaves 31
        --lgbm_max_depth -1
        --lgbm_subsample 0.8
        --lgbm_colsample_bytree 0.8
        --lgbm_min_child_samples 10
        --n_jobs 6
      )
      ETR_ARGS=(
        --etr_n_estimators 800
        --etr_max_depth 30
        --etr_max_features 0.2
        --etr_min_samples_leaf 3
        --etr_min_samples_split 8
        --n_jobs 6
      )
      ;;

    "GlassTransitionTemperature")
      CAT_ARGS=(
        --cat_iterations 600
        --cat_learning_rate 0.05
        --cat_depth 8
        --cat_l2_leaf_reg 3
        --cat_subsample 0.85
        --n_jobs 6
      )
      LGBM_ARGS=(
        --lgbm_n_estimators 400
        --lgbm_learning_rate 0.03
        --lgbm_num_leaves 31
        --lgbm_max_depth -1
        --lgbm_subsample 0.8
        --lgbm_colsample_bytree 0.8
        --lgbm_min_child_samples 20
        --n_jobs 6
      )
      ETR_ARGS=(
        --etr_n_estimators 800
        --etr_max_depth 30
        --etr_max_features 0.2
        --etr_min_samples_leaf 3
        --etr_min_samples_split 8
        --n_jobs 6
      )
      ;;

    "ElongationBreak")
      CAT_ARGS=(
        --cat_iterations 1000
        --cat_learning_rate 0.05
        --cat_depth 8
        --cat_l2_leaf_reg 3
        --cat_subsample 0.8
        --n_jobs 6
      )
      LGBM_ARGS=(
        --lgbm_n_estimators 400
        --lgbm_learning_rate 0.03
        --lgbm_num_leaves 31
        --lgbm_max_depth -1
        --lgbm_subsample 0.8
        --lgbm_colsample_bytree 0.8
        --lgbm_min_child_samples 10
        --n_jobs 6
      )
      ETR_ARGS=(
        --etr_n_estimators 800
        --etr_max_depth 30
        --etr_max_features 0.2
        --etr_min_samples_leaf 3
        --etr_min_samples_split 8
        --n_jobs 6
      )
      ;;

    "YoungsModulus")
      CAT_ARGS=(
        --cat_iterations 1000
        --cat_learning_rate 0.05
        --cat_depth 8
        --cat_l2_leaf_reg 5
        --cat_subsample 0.8
        --n_jobs 6
      )
      LGBM_ARGS=(
        --lgbm_n_estimators 400
        --lgbm_learning_rate 0.03
        --lgbm_num_leaves 64
        --lgbm_max_depth -1
        --lgbm_subsample 0.8
        --lgbm_colsample_bytree 0.8
        --lgbm_min_child_samples 10
        --n_jobs 6
      )
      ETR_ARGS=(
        --etr_n_estimators 400
        --etr_max_depth 30
        --etr_max_features 0.2
        --etr_min_samples_leaf 1
        --etr_min_samples_split 8
        --n_jobs 6
      )
      ;;

    "TensileStrength")
      CAT_ARGS=(
        --cat_iterations 600
        --cat_learning_rate 0.05
        --cat_depth 8
        --cat_l2_leaf_reg 3
        --cat_subsample 0.8
        --n_jobs 6
      )
      LGBM_ARGS=(
        --lgbm_n_estimators 400
        --lgbm_learning_rate 0.03
        --lgbm_num_leaves 31
        --lgbm_max_depth -1
        --lgbm_subsample 0.8
        --lgbm_colsample_bytree 1.0
        --lgbm_min_child_samples 10
        --n_jobs 6
      )
      ETR_ARGS=(
        --etr_n_estimators 800
        --etr_max_depth 30
        --etr_max_features 0.2
        --etr_min_samples_leaf 3
        --etr_min_samples_split 8
        --n_jobs 6
      )
      ;;

    *)
      echo "[ERROR] Unknown task name: $NAME"
      exit 1
      ;;
  esac

  # ===== CatBoost =====
  echo "[$(date)] Running CatBoost..."
  $PYTHON "$SCRIPT" \
    --in_csv "$INPUT" \
    --target_col "$TARGET" \
    --models catboost \
    "${COMMON_ARGS[@]}" \
    "${CAT_ARGS[@]}" \
    --out_dir "$OUT_CAT"

  # ===== LightGBM =====
  echo "[$(date)] Running LightGBM..."
  $PYTHON "$SCRIPT" \
    --in_csv "$INPUT" \
    --target_col "$TARGET" \
    --models lgbm \
    "${COMMON_ARGS[@]}" \
    "${LGBM_ARGS[@]}" \
    --out_dir "$OUT_LGBM"

  # ===== ExtraTrees =====
  echo "[$(date)] Running ExtraTrees..."
  $PYTHON "$SCRIPT" \
    --in_csv "$INPUT" \
    --target_col "$TARGET" \
    --models extratrees \
    "${COMMON_ARGS[@]}" \
    "${ETR_ARGS[@]}" \
    --out_dir "$OUT_ETR"

  echo "[$(date)] Finished ${NAME}"
done

echo "[$(date)] All 5 tasks completed."
