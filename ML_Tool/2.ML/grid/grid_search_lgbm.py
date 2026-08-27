# -*- coding: utf-8 -*-

import argparse
import itertools
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM 回归手写网格搜索（绕开 GridSearchCV 的 sklearn tags 兼容问题）")
    parser.add_argument("--in_csv", type=str, required=True, help="输入 CSV（包含特征与目标列）")
    parser.add_argument("--target_col", type=str, required=True, help="目标列名")
    parser.add_argument("--encoding", type=str, default=None, help="CSV 编码（默认自动尝试 utf-8-sig / gb18030 / latin1）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--cv_folds", type=int, default=10, help="KFold 折数，建议 >= 2")
    parser.add_argument("--n_jobs", type=int, default=-1, help="LightGBM 训练线程数")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--save_model", action="store_true", help="是否保存最终 best model")
    parser.add_argument(
        "--grid_preset",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="网格规模：small 更稳妥，medium 更大",
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


def extract_xy(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"目标列不存在: {target_col}")

    y = df[target_col].values
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values

    print(f"[INFO] 特征维度: {X.shape[1]}，样本数: {X.shape[0]}")
    print(f"[INFO] 目标列: {target_col}")
    return X, y, feature_names


def build_model(n_jobs, random_state, **params):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print("[ERROR] 未安装 lightgbm，请先运行: pip install lightgbm", file=sys.stderr)
        return None

    return LGBMRegressor(
        objective="regression",
        n_jobs=n_jobs,
        random_state=random_state,
        **params,
    )


def get_param_grid(preset):
    if preset == "medium":
        return {
            "n_estimators": [300, 600, 900],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [31, 64, 127],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_samples": [10, 20, 40],
        }

    return {
        "n_estimators": [400, 800],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [31, 64],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_samples": [10, 20],
    }


def ensure_out_dir(out_dir):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def save_feature_importance(model, feature_names, out_dir):
    if out_dir is None or not hasattr(model, "feature_importances_"):
        return
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": np.asarray(model.feature_importances_, dtype=float),
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)


def main():
    args = parse_args()

    if args.cv_folds < 2:
        raise ValueError("网格搜索建议使用 cv_folds >= 2")

    ensure_out_dir(args.out_dir)

    df = smart_read_csv(args.in_csv, encoding=args.encoding)
    X, y, feature_names = extract_xy(df, args.target_col)

    param_grid = list(ParameterGrid(get_param_grid(args.grid_preset)))
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    print("\n========== [LightGBM Manual Grid Search] ==========")
    print(f"[INFO] grid_preset = {args.grid_preset}")
    print(f"[INFO] cv_folds    = {args.cv_folds}")
    print(f"[INFO] 组合数      = {len(param_grid)}")

    all_rows = []
    best_params = None
    best_rmse = float("inf")

    for i, params in enumerate(param_grid, start=1):
        fold_r2, fold_rmse, fold_mae = [], [], []
        print(f"\n[INFO] 组合 {i}/{len(param_grid)}: {params}")

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = build_model(args.n_jobs, args.random_state, **params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)

            r2_v = r2_score(y_va, y_pred)
            rmse_v = rmse(y_va, y_pred)
            mae_v = mean_absolute_error(y_va, y_pred)

            fold_r2.append(r2_v)
            fold_rmse.append(rmse_v)
            fold_mae.append(mae_v)

            print(f"  Fold {fold}: R2={r2_v:.4f}, RMSE={rmse_v:.4f}, MAE={mae_v:.4f}")

        row = {
            **params,
            "mean_r2": float(np.mean(fold_r2)),
            "std_r2": float(np.std(fold_r2)),
            "mean_rmse": float(np.mean(fold_rmse)),
            "std_rmse": float(np.std(fold_rmse)),
            "mean_mae": float(np.mean(fold_mae)),
            "std_mae": float(np.std(fold_mae)),
        }
        all_rows.append(row)

        print(
            f"[INFO] mean: R2={row['mean_r2']:.4f}, RMSE={row['mean_rmse']:.4f}, MAE={row['mean_mae']:.4f}"
        )

        if row["mean_rmse"] < best_rmse:
            best_rmse = row["mean_rmse"]
            best_params = params.copy()

    print("\n[INFO] Best Params:")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print(f"[INFO] Best CV RMSE = {best_rmse:.6f}")

    final_model = build_model(args.n_jobs, args.random_state, **best_params)
    final_model.fit(X, y)

    summary = {
        "model": "lgbm",
        "search_mode": "manual_parameter_grid",
        "target_col": args.target_col,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_folds": int(args.cv_folds),
        "grid_preset": args.grid_preset,
        "best_params": best_params,
        "best_cv_rmse": float(best_rmse),
    }

    if args.out_dir is not None:
        with open(os.path.join(args.out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        pd.DataFrame(all_rows).sort_values("mean_rmse", ascending=True).to_csv(
            os.path.join(args.out_dir, "cv_results.csv"), index=False
        )

        save_feature_importance(final_model, feature_names, args.out_dir)

        if args.save_model:
            joblib.dump(final_model, os.path.join(args.out_dir, "model.pkl"))
            print(f"[INFO] 模型已保存到 {os.path.join(args.out_dir, 'model.pkl')}")

        print(f"[INFO] 网格搜索结果已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
