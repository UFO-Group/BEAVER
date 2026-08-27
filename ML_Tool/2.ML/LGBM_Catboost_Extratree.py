# -*- coding: utf-8 -*-

import argparse
import sys
import math
import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 LightGBM / CatBoost / ExtraTrees 对指纹特征做回归"
    )

    parser.add_argument("--in_csv", type=str, required=True, help="输入 CSV（包含特征与目标列）")
    parser.add_argument("--target_col", type=str, required=True, help="目标列名")

    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="CSV 编码（默认自动尝试 utf-8-sig / gb18030 / latin1）",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="测试集比例（默认 0.2，仅在 cv_folds <= 1 时使用）",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgbm", "catboost", "extratrees", "all"],
        default=["all"],
        help="要训练的模型：lgbm / catboost / extratrees / all（默认 all）",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="并行线程数（主要给 ExtraTrees / LightGBM 用）",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="保存模型与指标的输出目录（可选）",
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="是否保存训练好的模型到 out_dir",
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="K 折交叉验证折数；cv_folds <= 1 时使用单次 train_test_split，cv_folds >= 2 时使用 KFold CV",
    )

    # ================== LightGBM 超参数 ==================
    parser.add_argument("--lgbm_n_estimators", type=int, default=800)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--lgbm_num_leaves", type=int, default=64)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1)
    parser.add_argument("--lgbm_subsample", type=float, default=0.8)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=20)

    # ================== CatBoost 超参数 ==================
    parser.add_argument("--cat_iterations", type=int, default=1000)
    parser.add_argument("--cat_learning_rate", type=float, default=0.05)
    parser.add_argument("--cat_depth", type=int, default=8)
    parser.add_argument("--cat_l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--cat_subsample", type=float, default=1.0)

    # ================== ExtraTrees 超参数 ==================
    parser.add_argument("--etr_n_estimators", type=int, default=600)
    parser.add_argument("--etr_max_depth", type=int, default=-1)   # -1 表示 None
    parser.add_argument("--etr_max_features", type=float, default=0.2)
    parser.add_argument("--etr_min_samples_leaf", type=int, default=3)
    parser.add_argument("--etr_min_samples_split", type=int, default=8)

    return parser.parse_args()


def smart_read_csv(path, encoding=None):
    if encoding is not None:
        print(f"[INFO] 使用指定编码读取 CSV: {encoding}")
        return pd.read_csv(path, encoding=encoding)

    tried = []
    for enc in ["utf-8-sig", "gb18030", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] 成功读取文件，编码方式: {enc}")
            return df
        except Exception as e:
            tried.append((enc, str(e)))
    msg = "[ERROR] 无法读取 CSV 文件，请检查编码！尝试过的编码：\n"
    for enc, err in tried:
        msg += f"  - {enc}: {err}\n"
    raise RuntimeError(msg)


def train_test_split_xy(df, target_col, test_size, random_state):
    if target_col not in df.columns:
        raise ValueError(f"目标列不存在: {target_col}")

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    print(f"[INFO] 特征维度: {X.shape[1]}，样本数: {X.shape[0]}")
    print(f"[INFO] 目标列: {target_col}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"[INFO] 训练集: {X_train.shape[0]}，测试集: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def extract_xy(df, target_col):
    """用于 CV：不做划分，只返回全体 X, y 与特征名"""
    if target_col not in df.columns:
        raise ValueError(f"目标列不存在: {target_col}")

    y = df[target_col].values
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values

    print(f"[INFO] 特征维度: {X.shape[1]}，样本数: {X.shape[0]}")
    print(f"[INFO] 目标列: {target_col}")
    return X, y, feature_names


def eval_regression(y_true, y_pred, prefix=""):
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{prefix}R²   = {r2:.4f}")
    print(f"{prefix}RMSE = {rmse:.4f}")
    print(f"{prefix}MAE  = {mae:.4f}")
    return r2, rmse, mae


def get_model_dir(model_name, out_dir):
    if out_dir is None:
        return None
    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def build_prediction_df(X_part, y_true, y_pred, feature_names, target_col, split_name, fold=None):
    pred_df = pd.DataFrame(X_part, columns=feature_names)
    pred_df.insert(0, "split", split_name)
    if fold is not None:
        pred_df.insert(0, "fold", fold)
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["residual"] = np.asarray(y_true) - np.asarray(y_pred)
    if target_col not in pred_df.columns:
        pred_df[target_col] = y_true
    return pred_df


def save_prediction_csv(model_name, out_dir, filename, X_part, y_true, y_pred, feature_names, target_col, split_name, fold=None):
    model_dir = get_model_dir(model_name, out_dir)
    if model_dir is None:
        return

    pred_dir = os.path.join(model_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    pred_df = build_prediction_df(
        X_part=X_part,
        y_true=y_true,
        y_pred=y_pred,
        feature_names=feature_names,
        target_col=target_col,
        split_name=split_name,
        fold=fold,
    )
    pred_path = os.path.join(pred_dir, filename)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] {model_name}: 预测结果已保存到 {pred_path}")


def save_results(model_name, model, metrics, feature_names, out_dir, save_model):
    """
    将每个模型的结果保存到 out_dir/model_name/ 下：
      - metrics.json
      - feature_importances.csv（若支持）
      - model.pkl（若 save_model=True）
      - predictions/*.csv（训练/测试或每折 train/test 预测结果）
    """
    model_dir = get_model_dir(model_name, out_dir)
    if model_dir is None:
        return

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] {model_name}: 指标已保存到 {metrics_path}")

    if hasattr(model, "feature_importances_"):
        fi = np.array(model.feature_importances_, dtype=float)
        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": fi}
        ).sort_values("importance", ascending=False)
        fi_path = os.path.join(model_dir, "feature_importances.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"[INFO] {model_name}: 特征重要性已保存到 {fi_path}")

    if save_model:
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        print(f"[INFO] {model_name}: 模型已保存到 {model_path}")


def save_cv_fold_model(model_name, out_dir, fold, model):
    """保存 CV 每一折的模型。"""
    model_dir = get_model_dir(model_name, out_dir)
    if model_dir is None:
        return None

    fold_model_dir = os.path.join(model_dir, "fold_models")
    os.makedirs(fold_model_dir, exist_ok=True)

    model_path = os.path.join(fold_model_dir, f"fold_{fold}_model.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] {model_name}: 第 {fold} 折模型已保存到 {model_path}")
    return model_path


# ================== LightGBM ==================
def build_lgbm(args):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print("[ERROR] 未安装 lightgbm，请先运行: pip install lightgbm", file=sys.stderr)
        return None

    model = LGBMRegressor(
        n_estimators=args.lgbm_n_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        max_depth=args.lgbm_max_depth,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        min_child_samples=args.lgbm_min_child_samples,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    return model


def run_lgbm_holdout(X_train, X_test, y_train, y_test, args, out_dir, save_model, feature_names, target_col):
    model = build_lgbm(args)
    if model is None:
        return

    print("\n========== [LightGBM - Holdout] ==========")

    model.fit(X_train, y_train)

    print("[LightGBM] 训练集表现：")
    y_pred_train = model.predict(X_train)
    r2_tr, rmse_tr, mae_tr = eval_regression(y_train, y_pred_train, prefix="  ")

    print("[LightGBM] 测试集表现：")
    y_pred_test = model.predict(X_test)
    r2_te, rmse_te, mae_te = eval_regression(y_test, y_pred_test, prefix="  ")

    save_prediction_csv(
        "lgbm", out_dir, "holdout_train_predictions.csv",
        X_train, y_train, y_pred_train, feature_names, target_col,
        split_name="train",
    )
    save_prediction_csv(
        "lgbm", out_dir, "holdout_test_predictions.csv",
        X_test, y_test, y_pred_test, feature_names, target_col,
        split_name="test",
    )

    metrics = {
        "mode": "holdout",
        "params": model.get_params(),
        "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
        "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
    }
    save_results("lgbm", model, metrics, feature_names, out_dir, save_model)


def run_lgbm_cv(X, y, feature_names, cv_folds, args, out_dir, save_model, target_col):
    print(f"\n========== [LightGBM - {cv_folds}-Fold CV] ==========")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n[LightGBM][Fold {fold}/{cv_folds}]")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = build_lgbm(args)
        model.fit(X_tr, y_tr)

        print("  训练集：")
        y_pred_tr = model.predict(X_tr)
        r2_tr, rmse_tr, mae_tr = eval_regression(y_tr, y_pred_tr, prefix="    ")

        print("  测试集：")
        y_pred_te = model.predict(X_te)
        r2_te, rmse_te, mae_te = eval_regression(y_te, y_pred_te, prefix="    ")

        save_prediction_csv(
            "lgbm", out_dir, f"fold_{fold}_train_predictions.csv",
            X_tr, y_tr, y_pred_tr, feature_names, target_col,
            split_name="train", fold=fold,
        )
        save_prediction_csv(
            "lgbm", out_dir, f"fold_{fold}_test_predictions.csv",
            X_te, y_te, y_pred_te, feature_names, target_col,
            split_name="test", fold=fold,
        )

        fold_metrics.append({
            "fold": fold,
            "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
            "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
        })

    r2_vals = [fm["test"]["r2"] for fm in fold_metrics]
    rmse_vals = [fm["test"]["rmse"] for fm in fold_metrics]
    mae_vals = [fm["test"]["mae"] for fm in fold_metrics]

    cv_mean = {
        "r2": float(np.mean(r2_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "mae": float(np.mean(mae_vals)),
    }
    cv_std = {
        "r2": float(np.std(r2_vals)),
        "rmse": float(np.std(rmse_vals)),
        "mae": float(np.std(mae_vals)),
    }

    print("\n[LightGBM] CV 汇总（基于每折测试集）：")
    print(f"  R²   = {cv_mean['r2']:.4f} ± {cv_std['r2']:.4f}")
    print(f"  RMSE = {cv_mean['rmse']:.4f} ± {cv_std['rmse']:.4f}")
    print(f"  MAE  = {cv_mean['mae']:.4f} ± {cv_std['mae']:.4f}")

    final_model = build_lgbm(args)
    final_model.fit(X, y)

    metrics = {
        "mode": "cv",
        "params": final_model.get_params(),
        "cv_folds": cv_folds,
        "folds": fold_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }
    save_results("lgbm", final_model, metrics, feature_names, out_dir, save_model)


# ================== CatBoost ==================
def build_catboost(args):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("[ERROR] 未安装 catboost，请先运行: pip install catboost", file=sys.stderr)
        return None

    subsample = args.cat_subsample
    if subsample >= 1.0:
        subsample = None

    kwargs = dict(
        depth=args.cat_depth,
        learning_rate=args.cat_learning_rate,
        iterations=args.cat_iterations,
        loss_function="RMSE",
        random_seed=args.random_state,
        verbose=False,
        l2_leaf_reg=args.cat_l2_leaf_reg,
    )

    if subsample is not None:
        kwargs["subsample"] = subsample

    if args.n_jobs is not None and args.n_jobs > 0:
        kwargs["thread_count"] = args.n_jobs

    model = CatBoostRegressor(**kwargs)
    return model


def run_catboost_holdout(X_train, X_test, y_train, y_test, args, out_dir, save_model, feature_names, target_col):
    model = build_catboost(args)
    if model is None:
        return

    print("\n========== [CatBoost - Holdout] ==========")

    model.fit(X_train, y_train)

    print("[CatBoost] 训练集表现：")
    y_pred_train = model.predict(X_train)
    r2_tr, rmse_tr, mae_tr = eval_regression(y_train, y_pred_train, prefix="  ")

    print("[CatBoost] 测试集表现：")
    y_pred_test = model.predict(X_test)
    r2_te, rmse_te, mae_te = eval_regression(y_test, y_pred_test, prefix="  ")

    save_prediction_csv(
        "catboost", out_dir, "holdout_train_predictions.csv",
        X_train, y_train, y_pred_train, feature_names, target_col,
        split_name="train",
    )
    save_prediction_csv(
        "catboost", out_dir, "holdout_test_predictions.csv",
        X_test, y_test, y_pred_test, feature_names, target_col,
        split_name="test",
    )

    metrics = {
        "mode": "holdout",
        "params": model.get_params(),
        "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
        "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
    }
    save_results("catboost", model, metrics, feature_names, out_dir, save_model)


def run_catboost_cv(X, y, feature_names, cv_folds, args, out_dir, save_model, target_col):
    print(f"\n========== [CatBoost - {cv_folds}-Fold CV] ==========")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n[CatBoost][Fold {fold}/{cv_folds}]")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = build_catboost(args)
        model.fit(X_tr, y_tr)

        print("  训练集：")
        y_pred_tr = model.predict(X_tr)
        r2_tr, rmse_tr, mae_tr = eval_regression(y_tr, y_pred_tr, prefix="    ")

        print("  测试集：")
        y_pred_te = model.predict(X_te)
        r2_te, rmse_te, mae_te = eval_regression(y_te, y_pred_te, prefix="    ")

        save_prediction_csv(
            "catboost", out_dir, f"fold_{fold}_train_predictions.csv",
            X_tr, y_tr, y_pred_tr, feature_names, target_col,
            split_name="train", fold=fold,
        )
        save_prediction_csv(
            "catboost", out_dir, f"fold_{fold}_test_predictions.csv",
            X_te, y_te, y_pred_te, feature_names, target_col,
            split_name="test", fold=fold,
        )

        fold_metrics.append({
            "fold": fold,
            "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
            "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
        })

    r2_vals = [fm["test"]["r2"] for fm in fold_metrics]
    rmse_vals = [fm["test"]["rmse"] for fm in fold_metrics]
    mae_vals = [fm["test"]["mae"] for fm in fold_metrics]

    cv_mean = {
        "r2": float(np.mean(r2_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "mae": float(np.mean(mae_vals)),
    }
    cv_std = {
        "r2": float(np.std(r2_vals)),
        "rmse": float(np.std(rmse_vals)),
        "mae": float(np.std(mae_vals)),
    }

    print("\n[CatBoost] CV 汇总（基于每折测试集）：")
    print(f"  R²   = {cv_mean['r2']:.4f} ± {cv_std['r2']:.4f}")
    print(f"  RMSE = {cv_mean['rmse']:.4f} ± {cv_std['rmse']:.4f}")
    print(f"  MAE  = {cv_mean['mae']:.4f} ± {cv_std['mae']:.4f}")

    final_model = build_catboost(args)
    final_model.fit(X, y)

    metrics = {
        "mode": "cv",
        "params": final_model.get_params(),
        "cv_folds": cv_folds,
        "folds": fold_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }
    save_results("catboost", final_model, metrics, feature_names, out_dir, save_model)


# ================== ExtraTrees ==================
def build_extratrees(args):
    max_depth = None if args.etr_max_depth == -1 else args.etr_max_depth

    model = ExtraTreesRegressor(
        n_estimators=args.etr_n_estimators,
        max_depth=max_depth,
        max_features=args.etr_max_features,
        min_samples_leaf=args.etr_min_samples_leaf,
        min_samples_split=args.etr_min_samples_split,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    return model


def run_extratrees_holdout(X_train, X_test, y_train, y_test, args, out_dir, save_model, feature_names, target_col):
    print("\n========== [ExtraTreesRegressor - Holdout] ==========")
    model = build_extratrees(args)
    model.fit(X_train, y_train)

    print("[ExtraTrees] 训练集表现：")
    y_pred_train = model.predict(X_train)
    r2_tr, rmse_tr, mae_tr = eval_regression(y_train, y_pred_train, prefix="  ")

    print("[ExtraTrees] 测试集表现：")
    y_pred_test = model.predict(X_test)
    r2_te, rmse_te, mae_te = eval_regression(y_test, y_pred_test, prefix="  ")

    save_prediction_csv(
        "extratrees", out_dir, "holdout_train_predictions.csv",
        X_train, y_train, y_pred_train, feature_names, target_col,
        split_name="train",
    )
    save_prediction_csv(
        "extratrees", out_dir, "holdout_test_predictions.csv",
        X_test, y_test, y_pred_test, feature_names, target_col,
        split_name="test",
    )

    metrics = {
        "mode": "holdout",
        "params": model.get_params(),
        "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
        "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
    }
    save_results("extratrees", model, metrics, feature_names, out_dir, save_model)


def run_extratrees_cv(X, y, feature_names, cv_folds, args, out_dir, save_model, target_col):
    print(f"\n========== [ExtraTreesRegressor - {cv_folds}-Fold CV] ==========")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

    fold_metrics = []
    fold_model_paths = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n[ExtraTrees][Fold {fold}/{cv_folds}]")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = build_extratrees(args)
        model.fit(X_tr, y_tr)

        print("  训练集：")
        y_pred_tr = model.predict(X_tr)
        r2_tr, rmse_tr, mae_tr = eval_regression(y_tr, y_pred_tr, prefix="    ")

        print("  测试集：")
        y_pred_te = model.predict(X_te)
        r2_te, rmse_te, mae_te = eval_regression(y_te, y_pred_te, prefix="    ")

        save_prediction_csv(
            "extratrees", out_dir, f"fold_{fold}_train_predictions.csv",
            X_tr, y_tr, y_pred_tr, feature_names, target_col,
            split_name="train", fold=fold,
        )
        save_prediction_csv(
            "extratrees", out_dir, f"fold_{fold}_test_predictions.csv",
            X_te, y_te, y_pred_te, feature_names, target_col,
            split_name="test", fold=fold,
        )

        if save_model:
            fold_model_path = save_cv_fold_model("extratrees", out_dir, fold, model)
        else:
            fold_model_path = None

        fold_record = {
            "fold": fold,
            "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
            "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
        }
        if fold_model_path is not None:
            fold_record["model_path"] = fold_model_path
            fold_model_paths.append(fold_model_path)

        fold_metrics.append(fold_record)

    r2_vals = [fm["test"]["r2"] for fm in fold_metrics]
    rmse_vals = [fm["test"]["rmse"] for fm in fold_metrics]
    mae_vals = [fm["test"]["mae"] for fm in fold_metrics]

    cv_mean = {
        "r2": float(np.mean(r2_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "mae": float(np.mean(mae_vals)),
    }
    cv_std = {
        "r2": float(np.std(r2_vals)),
        "rmse": float(np.std(rmse_vals)),
        "mae": float(np.std(mae_vals)),
    }

    print("\n[ExtraTrees] CV 汇总（基于每折测试集）：")
    print(f"  R²   = {cv_mean['r2']:.4f} ± {cv_std['r2']:.4f}")
    print(f"  RMSE = {cv_mean['rmse']:.4f} ± {cv_std['rmse']:.4f}")
    print(f"  MAE  = {cv_mean['mae']:.4f} ± {cv_std['mae']:.4f}")

    final_model = build_extratrees(args)
    final_model.fit(X, y)

    metrics = {
        "mode": "cv",
        "params": final_model.get_params(),
        "cv_folds": cv_folds,
        "folds": fold_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }
    if save_model:
        metrics["fold_model_dir"] = os.path.join(get_model_dir("extratrees", out_dir), "fold_models")
        metrics["fold_model_paths"] = fold_model_paths

    save_results("extratrees", final_model, metrics, feature_names, out_dir, save_model)


def main():
    args = parse_args()

    df = smart_read_csv(args.in_csv, encoding=args.encoding)

    models = set(args.models)
    if "all" in models:
        models = {"lgbm", "catboost", "extratrees"}

    if args.out_dir is not None:
        print(f"[INFO] 结果将保存到目录: {args.out_dir}")

    print("[INFO] 当前模型超参数配置：")
    print(f"  [LGBM] n_estimators={args.lgbm_n_estimators}, learning_rate={args.lgbm_learning_rate}, "
          f"num_leaves={args.lgbm_num_leaves}, max_depth={args.lgbm_max_depth}, "
          f"subsample={args.lgbm_subsample}, colsample_bytree={args.lgbm_colsample_bytree}, "
          f"min_child_samples={args.lgbm_min_child_samples}")
    print(f"  [CatBoost] iterations={args.cat_iterations}, learning_rate={args.cat_learning_rate}, "
          f"depth={args.cat_depth}, l2_leaf_reg={args.cat_l2_leaf_reg}, subsample={args.cat_subsample}")
    print(f"  [ExtraTrees] n_estimators={args.etr_n_estimators}, max_depth={args.etr_max_depth}, "
          f"max_features={args.etr_max_features}, min_samples_leaf={args.etr_min_samples_leaf}, "
          f"min_samples_split={args.etr_min_samples_split}")

    use_cv = args.cv_folds is not None and args.cv_folds >= 2

    if use_cv:
        print(f"[INFO] 使用 {args.cv_folds}-Fold KFold 交叉验证")
        X, y, feat_names = extract_xy(df, target_col=args.target_col)

        if "lgbm" in models:
            run_lgbm_cv(
                X, y, feat_names,
                cv_folds=args.cv_folds,
                args=args,
                out_dir=args.out_dir,
                save_model=args.save_model,
                target_col=args.target_col,
            )

        if "catboost" in models:
            run_catboost_cv(
                X, y, feat_names,
                cv_folds=args.cv_folds,
                args=args,
                out_dir=args.out_dir,
                save_model=args.save_model,
                target_col=args.target_col,
            )

        if "extratrees" in models:
            run_extratrees_cv(
                X, y, feat_names,
                cv_folds=args.cv_folds,
                args=args,
                out_dir=args.out_dir,
                save_model=args.save_model,
                target_col=args.target_col,
            )

    else:
        print(f"[INFO] 使用单次 train_test_split，test_size={args.test_size}")
        X_train, X_test, y_train, y_test, feat_names = train_test_split_xy(
            df,
            target_col=args.target_col,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        if "lgbm" in models:
            run_lgbm_holdout(
                X_train, X_test, y_train, y_test,
                args=args,
                out_dir=args.out_dir,
                save_model=args.save_model,
                feature_names=feat_names,
                target_col=args.target_col,
            )

        if "catboost" in models:
            run_catboost_holdout(
                X_train, X_test, y_train, y_test,
                args=args,
                out_dir=args.out_dir,
                save_model=args.save_model,
                feature_names=feat_names,
                target_col=args.target_col,
            )

        if "extratrees" in models:
            run_extratrees_holdout(
                X_train, X_test, y_train, y_test,
                args=args,
                out_dir=args.out_dir,
                save_model=args.save_model,
                feature_names=feat_names,
                target_col=args.target_col,
            )


if __name__ == "__main__":
    main()
