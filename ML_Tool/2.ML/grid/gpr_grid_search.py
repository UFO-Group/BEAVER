# -*- coding: utf-8 -*-

import argparse
import json
import math
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# 通用工具
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Gaussian Process 回归网格搜索脚本")
    parser.add_argument("--in_csv", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument("--target_col", type=str, required=True, help="目标列名")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--encoding", type=str, default=None, help="CSV 编码，默认自动尝试")
    parser.add_argument("--cv_folds", type=int, default=5, help="KFold 折数，默认 5；GPR 不建议太大")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--n_jobs", type=int, default=4, help="GridSearchCV 并行数")
    parser.add_argument(
        "--refit_metric",
        type=str,
        default="neg_rmse",
        choices=["r2", "neg_rmse", "neg_mae"],
        help="用哪个指标选最佳模型，默认 neg_rmse",
    )
    parser.add_argument("--save_model", action="store_true", help="是否保存最佳模型")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="可选：若样本数过大，随机下采样到该数量后再做网格搜索，例如 800/1000/1500",
    )
    return parser.parse_args()



def smart_read_csv(path, encoding=None):
    if encoding is not None:
        print(f"[INFO] 使用指定编码读取 CSV: {encoding}")
        return pd.read_csv(path, encoding=encoding)

    tried = []
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] 成功读取文件，编码方式: {enc}")
            return df
        except Exception as e:
            tried.append((enc, str(e)))

    msg = "[ERROR] 无法读取 CSV 文件，请检查编码。尝试过的编码如下：\n"
    for enc, err in tried:
        msg += f"  - {enc}: {err}\n"
    raise RuntimeError(msg)



def prepare_xy(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"目标列不存在: {target_col}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")

    valid_mask = y.notna() & X.notna().all(axis=1)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"[WARN] 因目标列/特征中存在非数值或缺失，共删除 {dropped} 行")

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    print(f"[INFO] 样本数: {len(y)}")
    print(f"[INFO] 特征维度: {X.shape[1]}")
    print(f"[INFO] 目标列: {target_col}")

    return X, y, X.columns.tolist()



def maybe_subsample(X, y, max_samples, random_state):
    if max_samples is None or len(y) <= max_samples:
        return X, y, False

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    idx = np.sort(idx)
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y.iloc[idx].reset_index(drop=True)
    print(f"[WARN] GPR 复杂度较高，已从 {len(y)} 个样本下采样到 {len(y_sub)} 个样本用于网格搜索")
    return X_sub, y_sub, True



def rmse_func(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))



def eval_regression(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse_func(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }



def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# 主流程
# =========================
def build_kernel_candidates():
    return [
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),

        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),

        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),
    ]



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = smart_read_csv(args.in_csv, args.encoding)
    X, y, feature_names = prepare_xy(df, args.target_col)
    X_used, y_used, used_subsample = maybe_subsample(X, y, args.max_samples, args.random_state)

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    scoring = {
        "r2": "r2",
        "neg_rmse": make_scorer(rmse_func, greater_is_better=False),
        "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(random_state=args.random_state)),
    ])

    param_grid = {
        "model__kernel": build_kernel_candidates(),
        "model__alpha": [1e-8, 1e-6, 1e-4],
        "model__normalize_y": [True],
        "model__n_restarts_optimizer": [0, 1],
    }

    print("[INFO] 开始 GPR 网格搜索")
    print("[WARN] GPR 对样本量很敏感；若样本较大，建议把 cv_folds 调低到 3~5，并设置 --max_samples")
    print(f"[INFO] CV folds = {args.cv_folds}")
    print(f"[INFO] refit_metric = {args.refit_metric}")

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit=args.refit_metric,
        cv=cv,
        n_jobs=args.n_jobs,
        verbose=2,
        return_train_score=True,
    )
    grid.fit(X_used, y_used)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score_raw = float(grid.best_score_)

    best_score_readable = best_score_raw
    if args.refit_metric in {"neg_rmse", "neg_mae"}:
        best_score_readable = -best_score_raw

    print("\n========== [GPR Grid Search Finished] ==========")
    print(f"[BEST PARAMS] {best_params}")
    print(f"[BEST {args.refit_metric}] raw = {best_score_raw:.6f}")
    print(f"[BEST {args.refit_metric}] readable = {best_score_readable:.6f}")

    y_pred_used = best_model.predict(X_used)
    used_fit_metrics = eval_regression(y_used, y_pred_used)

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results_path = os.path.join(args.out_dir, "cv_results.csv")
    cv_results.to_csv(cv_results_path, index=False, encoding="utf-8-sig")

    summary = {
        "model": "gpr",
        "input_csv": args.in_csv,
        "target_col": args.target_col,
        "n_samples_original": int(len(y)),
        "n_samples_used_for_grid": int(len(y_used)),
        "used_subsample": bool(used_subsample),
        "max_samples": args.max_samples,
        "n_features": int(X.shape[1]),
        "cv_folds": int(args.cv_folds),
        "random_state": int(args.random_state),
        "refit_metric": args.refit_metric,
        "best_score_raw": best_score_raw,
        "best_score_readable": best_score_readable,
        "best_params": {k: str(v) if k == "model__kernel" else v for k, v in best_params.items()},
        "used_fit_metrics": used_fit_metrics,
        "feature_names_file": "feature_names.json",
        "cv_results_file": "cv_results.csv",
    }

    save_json(os.path.join(args.out_dir, "best_summary.json"), summary)
    save_json(os.path.join(args.out_dir, "feature_names.json"), feature_names)

    if args.save_model:
        model_path = os.path.join(args.out_dir, "best_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"[INFO] 最佳模型已保存到: {model_path}")

    print(f"[INFO] cv_results 已保存到: {cv_results_path}")
    print(f"[INFO] summary 已保存到: {os.path.join(args.out_dir, 'best_summary.json')}")


if __name__ == "__main__":
    main()
