# Machine-Learning Property Regression

This folder contains the machine-learning workflow used to generate molecular features, optimize regression models, perform 10-fold cross-validation, draw performance figures, and predict polymer properties for new candidates.

## Workflow

1. Place the property tables containing polymer SMILES in `DataBase/0401/10/`.
2. Run `morgan_pooling_new.py` to convert `SMILE A`–`SMILE E` into pooled Morgan-fingerprint features.
3. Use the scripts in `grid/` to search the model hyperparameters.
4. Train the final models with 10-fold cross-validation and save their metrics, predictions, and fitted models under `Result/`.
5. Use the scripts in `draw/` to generate true-versus-predicted performance plots.
6. Use the scripts in `predict/` to apply selected trained models to new polymer candidates.

The workflow evaluates the following five properties:

| Folder name | Target column |
|---|---|
| `GlassTransitionTemperature` | `Glass Transition Temperature (°C)` |
| `MeltingTemperature` | `Melting Temperature (°C)` |
| `YoungsModulus` | `Young's Modulus (kPa) log10` |
| `TensileStrength` | `Tensile Strength (MPa) log10` |
| `ElongationBreak` | `Elongation at Break (%) log10` |

## Main files and folders

| File or folder | Purpose |
|---|---|
| `DataBase/0401/10/` | Original property tables and generated 4096-dimensional pooled-feature CSV files. |
| `morgan_pooling_new.py` | Generates pooled Morgan features from multiple polymer SMILES columns. With `nbits=2048`, the polymer and additive fingerprint blocks are concatenated into 4096 features. |
| `morgan_new.slurm` | Slurm wrapper for Morgan-feature generation. Enable or edit the required property entries before submission. |
| `baseline_OLD.py` | Trains RF and optional XGB/MLP/RF+MLP models; supports hold-out evaluation and K-fold cross-validation. |
| `baseline_xgboost_mlp_svm.py` | Trains XGBoost, SVM, MLP, RF, or RF+MLP models and exports fold-level results. |
| `LGBM_Catboost_Extratree.py` | Trains LightGBM, CatBoost, and ExtraTrees models using hold-out or K-fold evaluation. |
| `KNN_GP.py` | Trains K-nearest-neighbor and Gaussian-process regression models. |
| `grid/` | Hyperparameter-search scripts for RF, SVM, XGB, LightGBM, CatBoost, ExtraTrees, KNN, and GPR. |
| `Result/` | Final model files, fold predictions, feature importance, and evaluation metrics for the five properties. |
| `draw/` | Generates R² scatter plots and optional marginal-density plots from train/validation prediction tables. |
| `predict/` | Applies selected RF, XGB, or ExtraTrees models to pooled features for new candidates. |

The eight candidate models used for comparison are RF, SVM, XGB, ExtraTrees, LightGBM, CatBoost, KNN, and Gaussian process regression. MLP-related code is retained as an optional additional baseline.

## Installation

Python 3.9 is recommended.

```bash
pip install -r requirements.txt
```

The pinned PyTorch packages use CUDA 11.6 wheels. If the target system uses another CUDA version or CPU-only PyTorch, install the appropriate PyTorch build separately.

## Example: feature generation

Run commands from the `2.ML` directory. For example, Young's-modulus features can be generated with the fingerprint settings recorded in `morgan_new.slurm`:

```bash
python morgan_pooling_new.py \
  --in_csv DataBase/0401/10/YoungsModulus-log10.csv \
  --out_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --polymer_cols "SMILE A" "SMILE B" "SMILE C" "SMILE D" "SMILE E" \
  --target_cols "Young's Modulus (kPa) log10" \
  --radius 3 \
  --nbits 2048 \
  --feature_mode 2048 \
  --dr_method none
```

## Example: hyperparameter grid search

The following examples use the Young's-modulus feature table. Replace only the input file, target column, and output directory to optimize another property.

Random forest:

```bash
python grid/rf_grid_loop.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target "Young's Modulus (kPa) log10" \
  --save_dir grid/outputs/YoungsModulus/rf_grid_loop-YoungsModulus \
  --id_cols SampleID,RecipeID \
  --test_size 0.2
```

XGBoost:

```bash
python grid/grid_xgb.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target "Young's Modulus (kPa) log10" \
  --save_dir grid/outputs/YoungsModulus/xgb_grid_loop-YoungsModulus
```

Support vector machine:

```bash
python grid/grid_svm.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target "Young's Modulus (kPa) log10" \
  --save_dir grid/outputs/YoungsModulus/svm_grid_loop-YoungsModulus
```

LightGBM:

```bash
python grid/grid_search_lgbm.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --cv_folds 10 --n_jobs 8 --grid_preset small --save_model \
  --out_dir grid/outputs/YoungsModulus/YoungsModulus-lgbm_loop
```

CatBoost:

```bash
python grid/grid_search_catboost.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --cv_folds 10 --n_jobs 8 --grid_preset small --save_model \
  --out_dir grid/outputs/YoungsModulus/YoungsModulus-catboost_loop
```

ExtraTrees:

```bash
python grid/grid_search_extratrees.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --cv_folds 10 --n_jobs 8 --grid_preset small --save_model \
  --out_dir grid/outputs/YoungsModulus/YoungsModulus-extratrees_loop
```

KNN:

```bash
python grid/knn_grid_search.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --cv_folds 10 --random_state 42 --n_jobs 6 \
  --refit_metric neg_rmse --save_model \
  --out_dir grid/outputs/YoungsModulus/knn
```

Gaussian process regression:

```bash
python grid/gpr_grid_search.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --cv_folds 10 --random_state 42 --n_jobs 6 \
  --refit_metric neg_rmse --max_samples 1000 --save_model \
  --out_dir grid/outputs/YoungsModulus/gpr
```

GPR grid search is substantially more expensive than the other searches. The supplied Slurm job uses 10 folds and limits the search to 1,000 samples.

## Example: 10-fold model training

Random forest:

```bash
python baseline_OLD.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target "Young's Modulus (kPa) log10" \
  --model rf \
  --save_dir Result/YoungsModulus/rf_cv10 \
  --cv10 --cv_folds 10 --save_train_pred \
  --seed 42 \
  --rf_n_estimators 300 \
  --rf_max_depth 20 \
  --rf_min_samples_split 8 \
  --rf_min_samples_leaf 1 \
  --rf_max_features 0.1
```

XGBoost:

```bash
python baseline_xgboost_mlp_svm.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target "Young's Modulus (kPa) log10" \
  --model xgb \
  --save_dir Result/YoungsModulus/xgb \
  --cv10 --cv_folds 10 --save_train_pred \
  --seed 42 \
  --xgb_n_estimators 1000 \
  --xgb_learning_rate 0.01 \
  --xgb_max_depth 8 \
  --xgb_subsample 0.7 \
  --xgb_colsample_bytree 0.3 \
  --xgb_reg_lambda 3.0
```

Support vector machine:

```bash
python baseline_xgboost_mlp_svm.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target "Young's Modulus (kPa) log10" \
  --model svm \
  --save_dir Result/YoungsModulus/svm \
  --cv10 --cv_folds 10 --save_train_pred \
  --svm_kernel rbf \
  --svm_C 50.0 \
  --svm_epsilon 0.2 \
  --svm_gamma auto \
  --no_perm
```

LightGBM, CatBoost, and ExtraTrees:

```bash
python LGBM_Catboost_Extratree.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --models lgbm catboost extratrees \
  --cv_folds 10 --test_size 0.1 --random_state 42 --save_model \
  --n_jobs 6 \
  --cat_iterations 1000 --cat_learning_rate 0.05 \
  --cat_depth 8 --cat_l2_leaf_reg 5 --cat_subsample 0.8 \
  --lgbm_n_estimators 400 --lgbm_learning_rate 0.03 \
  --lgbm_num_leaves 64 --lgbm_max_depth -1 \
  --lgbm_subsample 0.8 --lgbm_colsample_bytree 0.8 \
  --lgbm_min_child_samples 10 \
  --etr_n_estimators 400 --etr_max_depth 30 \
  --etr_max_features 0.2 --etr_min_samples_leaf 1 \
  --etr_min_samples_split 8 \
  --out_dir Result/YoungsModulus/tree_models
```

KNN:

```bash
python KNN_GP.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --models knn \
  --cv_folds 10 --knn_n_neighbors 9 \
  --n_jobs 8 --random_state 42 --save_model \
  --out_dir Result/YoungsModulus/knn
```

Gaussian process regression:

```bash
python KNN_GP.py \
  --in_csv DataBase/0401/10/YoungsModulus-pooled-4096.csv \
  --target_col "Young's Modulus (kPa) log10" \
  --models gpr \
  --cv_folds 10 --random_state 42 --save_model \
  --out_dir Result/YoungsModulus/gpr
```

The principal evaluation outputs are `metrics.json`, fold-level prediction CSV files, cross-validation summaries, feature-importance tables where supported, and serialized model files (`.pkl` or `.joblib`).

## Example: prediction on new candidates

The original candidate table must first be converted with the same Morgan settings used for the training data. This local example contains two SMILES columns; the numerical fingerprint settings match `morgan_predict.slurm`.

```bash
python morgan_pooling_new.py \
  --in_csv predict/outputs/SMILEAB.csv \
  --out_csv predict/outputs/SMILEAB_morgan.csv \
  --polymer_cols "SMILE A" "SMILE B" \
  --target_cols "Polymer A" "Polymer B" \
  --radius 3 \
  --nbits 2048 \
  --feature_mode 2048 \
  --dr_method none
```

Random-forest prediction:

```bash
python predict/predict_rf.py \
  --in_csv predict/outputs/SMILEAB_morgan.csv \
  --source_csv predict/outputs/SMILEAB.csv \
  --out_csv predict/outputs/Result/Prediction-YoungsModulus-rf.csv \
  --model_dir Result/YoungsModulus/rf_cv10 \
  --target_name "Young's Modulus (kPa) log10"
```

XGBoost prediction:

```bash
python predict/predict_xgb.py \
  --in_csv predict/outputs/SMILEAB_morgan.csv \
  --source_csv predict/outputs/SMILEAB.csv \
  --out_csv predict/outputs/Result/Prediction-YoungsModulus-xgb.csv \
  --model_dir Result/YoungsModulus/xgb \
  --model_name best_model.joblib \
  --target_name "Young's Modulus (kPa) log10"
```

ExtraTrees prediction:

```bash
python predict/predict_ext.py \
  --in_csv predict/outputs/SMILEAB_morgan.csv \
  --source_csv predict/outputs/SMILEAB.csv \
  --out_csv predict/outputs/Result/Prediction-YoungsModulus-extratrees.csv \
  --model_dir Result/YoungsModulus/extratrees \
  --target_name "Young's Modulus (kPa) log10" \
  --id_col row_index
```

The model directory must contain `best_model.joblib` for RF/XGB or `model.pkl` for ExtraTrees. The source and pooled-feature tables should contain the same samples in the same order, or share the identifier specified with `--id_col`.

## Example: performance plotting

The following command reproduces the plotting parameters recorded in `draw_r2_joint.slurm`:

```bash
python draw/draw_r2_joint.py \
  --train_csv Result/TensileStrength/rf_cv10/fold_05_train.csv \
  --test_csv Result/TensileStrength/rf_cv10/fold_05_valid.csv \
  --outdir Result/TensileStrength/rf_cv10/draw_kde_attached \
  --linear_tick_nbins 6 \
  --log_major_step 0.5 \
  --bandwidth_scale 1.15 \
  --marginal_size 20%
```

The plotting scripts expect positive values on the original linear scale when generating their additional log10 panels. The Young's-modulus, tensile-strength, and elongation-at-break training targets in this folder are already stored as log10 values; do not apply a second log10 transformation to those columns. In addition, `draw_r2.py` contains property-specific hard-coded titles, so they should be adjusted before it is used for another property.
