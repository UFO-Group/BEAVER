# -*- coding: utf-8 -*-

import argparse
import json
import math
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# 通用工具
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="KNN 回归网格搜索脚本")
    parser.add_argument("--in_csv", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument("--target_col", type=str, required=True, help="目标列名")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--encoding", type=str, default=None, help="CSV 编码，默认自动尝试")
    parser.add_argument("--cv_folds", type=int, default=10, help="KFold 折数，默认 10")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--n_jobs", type=int, default=8, help="GridSearchCV 并行数")
    parser.add_argument(
        "--refit_metric",
        type=str,
        default="neg_rmse",
        choices=["r2", "neg_rmse", "neg_mae"],
        help="用哪个指标选最佳模型，默认 neg_rmse",
    )
    parser.add_argument("--save_model", action="store_true", help="是否保存最佳模型")
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
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = smart_read_csv(args.in_csv, args.encoding)
    X, y, feature_names = prepare_xy(df, args.target_col)

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    scoring = {
        "r2": "r2",
        "neg_rmse": make_scorer(rmse_func, greater_is_better=False),
        "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor()),
    ])

    param_grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
        "model__leaf_size": [20, 30, 40],
        "model__algorithm": ["auto"],
    }

    print("[INFO] 开始 KNN 网格搜索")
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
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score_raw = float(grid.best_score_)

    # 为了便于阅读，把负号指标翻回正数
    best_score_readable = best_score_raw
    if args.refit_metric in {"neg_rmse", "neg_mae"}:
        best_score_readable = -best_score_raw

    print("\n========== [KNN Grid Search Finished] ==========")
    print(f"[BEST PARAMS] {best_params}")
    print(f"[BEST {args.refit_metric}] raw = {best_score_raw:.6f}")
    print(f"[BEST {args.refit_metric}] readable = {best_score_readable:.6f}")

    # 在全数据上用 best estimator 预测，保存一个参考性的全量拟合指标
    y_pred_full = best_model.predict(X)
    full_fit_metrics = eval_regression(y, y_pred_full)

    # 保存 cv_results
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results_path = os.path.join(args.out_dir, "cv_results.csv")
    cv_results.to_csv(cv_results_path, index=False, encoding="utf-8-sig")

    summary = {
        "model": "knn",
        "input_csv": args.in_csv,
        "target_col": args.target_col,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "cv_folds": int(args.cv_folds),
        "random_state": int(args.random_state),
        "refit_metric": args.refit_metric,
        "best_score_raw": best_score_raw,
        "best_score_readable": best_score_readable,
        "best_params": best_params,
        "full_fit_metrics": full_fit_metrics,
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
