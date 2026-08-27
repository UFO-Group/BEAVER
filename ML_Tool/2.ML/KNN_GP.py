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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import joblib


# ================== 通用工具 ==================
def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 KNN / Gaussian Process 对指纹特征做回归"
    )

    parser.add_argument("--in_csv", type=str, required=True, help="输入 CSV（包含特征与目标列）")
    parser.add_argument("--target_col", type=str, required=True, help="目标列名（例如 'Melting Temperature (°C)'）")

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
        help="随机种子（默认 42，用于划分和 KFold）",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["knn", "gpr", "all"],
        default=["all"],
        help="要训练的模型：knn / gpr / all（默认 all）",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="并行线程数（对 KNN 有效；GPR 不支持 n_jobs）",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="保存模型与指标的输出目录（可选；每个模型会在该目录下建子目录）",
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="是否保存训练好的模型到 out_dir（默认不保存）",
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="K 折交叉验证折数；cv_folds <= 1 时使用单次 train_test_split，cv_folds >= 2 时使用 KFold CV",
    )

    # 可选：KNN 的邻居数
    parser.add_argument(
        "--knn_n_neighbors",
        type=int,
        default=15,
        help="KNN 的 k（默认 15）",
    )

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


def save_results(model_name, model, metrics, feature_names, out_dir, save_model):
    """
    将每个模型的结果保存到 out_dir/model_name/ 下：
      - metrics.json
      - （如果有）feature_importances.csv
      - model.pkl（若 save_model=True）
    """
    if out_dir is None:
        return

    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 保存 metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] {model_name}: 指标已保存到 {metrics_path}")

    # KNN / GPR 通常没有 feature_importances_，这里会自动跳过
    if hasattr(model, "feature_importances_"):
        fi = np.array(model.feature_importances_, dtype=float)
        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": fi}
        ).sort_values("importance", ascending=False)
        fi_path = os.path.join(model_dir, "feature_importances.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"[INFO] {model_name}: 特征重要性已保存到 {fi_path}")

    # 保存模型
    if save_model:
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        print(f"[INFO] {model_name}: 模型已保存到 {model_path}")


# ================== KNN ==================
def build_knn(n_neighbors, n_jobs):
    # 使用距离加权，通常比 uniform 略好
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="distance",
        n_jobs=n_jobs,
    )
    return model


def run_knn_holdout(X_train, X_test, y_train, y_test, n_neighbors, n_jobs, out_dir, save_model, feature_names):
    print("\n========== [KNN - Holdout] ==========")
    model = build_knn(n_neighbors, n_jobs)
    model.fit(X_train, y_train)

    print("[KNN] 训练集表现：")
    y_pred_tr = model.predict(X_train)
    r2_tr, rmse_tr, mae_tr = eval_regression(y_train, y_pred_tr, prefix="  ")

    print("[KNN] 测试集表现：")
    y_pred_te = model.predict(X_test)
    r2_te, rmse_te, mae_te = eval_regression(y_test, y_pred_te, prefix="  ")

    metrics = {
        "mode": "holdout",
        "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
        "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
    }
    save_results("knn", model, metrics, feature_names, out_dir, save_model)


def run_knn_cv(X, y, feature_names, cv_folds, n_neighbors, n_jobs, out_dir, save_model, random_state):
    print(f"\n========== [KNN - {cv_folds}-Fold CV] ==========")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n[KNN][Fold {fold}/{cv_folds}]")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_knn(n_neighbors, n_jobs)
        model.fit(X_tr, y_tr)

        print("  训练集：")
        y_pred_tr = model.predict(X_tr)
        r2_tr, rmse_tr, mae_tr = eval_regression(y_tr, y_pred_tr, prefix="    ")

        print("  验证集：")
        y_pred_val = model.predict(X_val)
        r2_val, rmse_val, mae_val = eval_regression(y_val, y_pred_val, prefix="    ")

        fold_metrics.append({
            "fold": fold,
            "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
            "val": {"r2": r2_val, "rmse": rmse_val, "mae": mae_val},
        })

    r2_vals = [fm["val"]["r2"] for fm in fold_metrics]
    rmse_vals = [fm["val"]["rmse"] for fm in fold_metrics]
    mae_vals = [fm["val"]["mae"] for fm in fold_metrics]

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

    print("\n[KNN] CV 汇总（基于验证集）：")
    print(f"  R²   = {cv_mean['r2']:.4f} ± {cv_std['r2']:.4f}")
    print(f"  RMSE = {cv_mean['rmse']:.4f} ± {cv_std['rmse']:.4f}")
    print(f"  MAE  = {cv_mean['mae']:.4f} ± {cv_std['mae']:.4f}")

    # final 模型：在全数据上训练一次，方便保存
    final_model = build_knn(n_neighbors, n_jobs)
    final_model.fit(X, y)

    metrics = {
        "mode": "cv",
        "cv_folds": cv_folds,
        "folds": fold_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }
    save_results("knn", final_model, metrics, feature_names, out_dir, save_model)


# ================== Gaussian Process ==================
def build_gpr():
    # 一个相对稳健的 kernel：RBF + WhiteKernel
    # 注意：在高维、大样本场景下会非常慢！
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=42,
    )
    return model


def run_gpr_holdout(X_train, X_test, y_train, y_test, out_dir, save_model, feature_names):
    print("\n========== [Gaussian Process - Holdout] ==========")
    print("[WARN] GPR 复杂度 ~ O(N^3)，大样本会非常慢，建议先在子集上测试。")

    model = build_gpr()
    model.fit(X_train, y_train)

    print("[GPR] 训练集表现：")
    y_pred_tr = model.predict(X_train)
    r2_tr, rmse_tr, mae_tr = eval_regression(y_train, y_pred_tr, prefix="  ")

    print("[GPR] 测试集表现：")
    y_pred_te = model.predict(X_test)
    r2_te, rmse_te, mae_te = eval_regression(y_test, y_pred_te, prefix="  ")

    metrics = {
        "mode": "holdout",
        "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
        "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
    }
    save_results("gpr", model, metrics, feature_names, out_dir, save_model)


def run_gpr_cv(X, y, feature_names, cv_folds, out_dir, save_model, random_state):
    print(f"\n========== [Gaussian Process - {cv_folds}-Fold CV] ==========")
    print("[WARN] GPR 复杂度 ~ O(N^3)，在几千样本上做 CV 可能非常慢甚至爆内存，请谨慎使用。")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n[GPR][Fold {fold}/{cv_folds}]")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_gpr()
        model.fit(X_tr, y_tr)

        print("  训练集：")
        y_pred_tr = model.predict(X_tr)
        r2_tr, rmse_tr, mae_tr = eval_regression(y_tr, y_pred_tr, prefix="    ")

        print("  验证集：")
        y_pred_val = model.predict(X_val)
        r2_val, rmse_val, mae_val = eval_regression(y_val, y_pred_val, prefix="    ")

        fold_metrics.append({
            "fold": fold,
            "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
            "val": {"r2": r2_val, "rmse": rmse_val, "mae": mae_val},
        })

    r2_vals = [fm["val"]["r2"] for fm in fold_metrics]
    rmse_vals = [fm["val"]["rmse"] for fm in fold_metrics]
    mae_vals = [fm["val"]["mae"] for fm in fold_metrics]

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

    print("\n[GPR] CV 汇总（基于验证集）：")
    print(f"  R²   = {cv_mean['r2']:.4f} ± {cv_std['r2']:.4f}")
    print(f"  RMSE = {cv_mean['rmse']:.4f} ± {cv_std['rmse']:.4f}")
    print(f"  MAE  = {cv_mean['mae']:.4f} ± {cv_std['mae']:.4f}")

    final_model = build_gpr()
    final_model.fit(X, y)

    metrics = {
        "mode": "cv",
        "cv_folds": cv_folds,
        "folds": fold_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }
    save_results("gpr", final_model, metrics, feature_names, out_dir, save_model)


# ================== main ==================
def main():
    args = parse_args()

    df = smart_read_csv(args.in_csv, encoding=args.encoding)

    # 解析 models 选项
    models = set(args.models)
    if "all" in models:
        models = {"knn", "gpr"}

    if args.out_dir is not None:
        print(f"[INFO] 结果将保存到目录: {args.out_dir}")

    use_cv = args.cv_folds is not None and args.cv_folds >= 2

    if use_cv:
        print(f"[INFO] 使用 {args.cv_folds}-Fold KFold 交叉验证")
        X, y, feat_names = extract_xy(df, target_col=args.target_col)

        if "knn" in models:
            run_knn_cv(
                X, y, feat_names,
                cv_folds=args.cv_folds,
                n_neighbors=args.knn_n_neighbors,
                n_jobs=args.n_jobs,
                out_dir=args.out_dir,
                save_model=args.save_model,
                random_state=args.random_state,
            )

        if "gpr" in models:
            run_gpr_cv(
                X, y, feat_names,
                cv_folds=args.cv_folds,
                out_dir=args.out_dir,
                save_model=args.save_model,
                random_state=args.random_state,
            )

    else:
        print(f"[INFO] 使用单次 train_test_split，test_size={args.test_size}")
        X_train, X_test, y_train, y_test, feat_names = train_test_split_xy(
            df,
            target_col=args.target_col,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        if "knn" in models:
            run_knn_holdout(
                X_train, X_test, y_train, y_test,
                n_neighbors=args.knn_n_neighbors,
                n_jobs=args.n_jobs,
                out_dir=args.out_dir,
                save_model=args.save_model,
                feature_names=feat_names,
            )

        if "gpr" in models:
            run_gpr_holdout(
                X_train, X_test, y_train, y_test,
                out_dir=args.out_dir,
                save_model=args.save_model,
                feature_names=feat_names,
            )


if __name__ == "__main__":
    main()
