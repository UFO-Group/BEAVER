# -*- coding: utf-8 -*-

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_validate


def parse_args():
    parser = argparse.ArgumentParser(description="ExtraTrees 回归网格搜索")
    parser.add_argument("--in_csv", type=str, required=True, help="输入 CSV（包含特征与目标列）")
    parser.add_argument("--target_col", type=str, required=True, help="目标列名")
    parser.add_argument("--encoding", type=str, default=None, help="CSV 编码（默认自动尝试 utf-8-sig / gb18030 / latin1）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--cv_folds", type=int, default=10, help="KFold 折数，建议 >= 2")
    parser.add_argument("--n_jobs", type=int, default=-1, help="ExtraTrees 训练线程数")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--save_model", action="store_true", help="是否保存最终 best model")
    parser.add_argument(
        "--grid_preset",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="网格规模：small 更稳妥，medium 更大",
    )
    parser.add_argument("--verbose", type=int, default=2, help="GridSearchCV verbose")
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


def build_model(n_jobs, random_state):
    return ExtraTreesRegressor(
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_param_grid(preset):
    if preset == "medium":
        return {
            "n_estimators": [300, 600, 900],
            "max_depth": [None, 20, 40],
            "max_features": [0.2, 0.4, "sqrt"],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 3],
        }

    return {
        "n_estimators": [400, 800],
        "max_depth": [None, 30],
        "max_features": [0.2, "sqrt"],
        "min_samples_split": [2, 8],
        "min_samples_leaf": [1, 3],
    }


def ensure_out_dir(out_dir):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)


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

    model = build_model(args.n_jobs, args.random_state)
    param_grid = get_param_grid(args.grid_preset)
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="rmse",
        cv=cv,
        n_jobs=1,
        verbose=args.verbose,
        return_train_score=False,
    )

    print("\n========== [ExtraTrees Grid Search] ==========")
    print(f"[INFO] grid_preset = {args.grid_preset}")
    print(f"[INFO] cv_folds    = {args.cv_folds}")
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_rmse = -float(grid.best_score_)

    print("\n[INFO] Best Params:")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print(f"[INFO] Best CV RMSE = {best_cv_rmse:.6f}")

    cv_eval = cross_validate(
        clone(best_model),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    summary = {
        "model": "extratrees",
        "target_col": args.target_col,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_folds": int(args.cv_folds),
        "grid_preset": args.grid_preset,
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "cv_summary": {
            "r2_mean": float(np.mean(cv_eval["test_r2"])),
            "r2_std": float(np.std(cv_eval["test_r2"])),
            "rmse_mean": float(-np.mean(cv_eval["test_rmse"])),
            "rmse_std": float(np.std(-cv_eval["test_rmse"])),
            "mae_mean": float(-np.mean(cv_eval["test_mae"])),
            "mae_std": float(np.std(-cv_eval["test_mae"])),
        },
    }

    print("\n[INFO] Best Model CV Summary:")
    print(json.dumps(summary["cv_summary"], indent=2, ensure_ascii=False))

    final_model = clone(best_model)
    final_model.fit(X, y)

    if args.out_dir is not None:
        with open(os.path.join(args.out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        results_df = pd.DataFrame(grid.cv_results_).copy()
        for col in ["mean_test_rmse", "std_test_rmse", "mean_test_mae", "std_test_mae"]:
            if col in results_df.columns:
                results_df[col] = -results_df[col]
        results_df.to_csv(os.path.join(args.out_dir, "cv_results.csv"), index=False)

        save_feature_importance(final_model, feature_names, args.out_dir)

        if args.save_model:
            joblib.dump(final_model, os.path.join(args.out_dir, "model.pkl"))
            print(f"[INFO] 模型已保存到 {os.path.join(args.out_dir, 'model.pkl')}")

        print(f"[INFO] 网格搜索结果已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
