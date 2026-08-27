#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 网格搜索 + 最优参数十折交叉验证评估（支持 SLURM 提交）
"""
import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import argparse

def main():
    # ========== 参数解析 ==========
    ap = argparse.ArgumentParser(description="XGBoost Grid Search + 10-Fold CV")
    ap.add_argument("--in_csv", required=True, help="输入特征CSV")
    ap.add_argument("--target", required=True, help="目标列名")
    ap.add_argument("--save_dir", default="xgb_grid", help="输出目录")
    args = ap.parse_args()

    in_csv = args.in_csv
    target = args.target
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========== 数据读取 ==========
    df = pd.read_csv(in_csv)
    X = df.select_dtypes(include=[np.number]).drop(columns=[target])
    y = df[target].values

    # ========== 参数网格 ==========
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [500, 800, 1000],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.3, 0.5],
        "min_child_weight": [3, 6, 10],
        "reg_lambda": [3, 6, 9]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"搜索参数组合数量: {len(list(ParameterGrid(param_grid)))}\n")

    # ========== 手动网格搜索 ==========
    results_list = []
    for i, params in enumerate(ParameterGrid(param_grid), 1):
        scores = []
        for train_idx, valid_idx in kf.split(X, y):
            Xtr, Xva = X.iloc[train_idx], X.iloc[valid_idx]
            ytr, yva = y[train_idx], y[valid_idx]

            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
                n_jobs=1,
                **params
            )
            model.fit(Xtr, ytr)
            y_pred = model.predict(Xva)
            r2 = r2_score(yva, y_pred)
            scores.append(r2)

        mean_r2 = np.mean(scores)
        results_list.append({**params, "mean_R2": mean_r2})
        print(f"[{i:03d}/{len(ParameterGrid(param_grid))}] mean_R2 = {mean_r2:.4f} | params = {params}")

    # ========== 保存搜索结果 ==========
    results_df = pd.DataFrame(results_list)
    best_row = results_df.loc[results_df["mean_R2"].idxmax()]
    best_params = {k: best_row[k] for k in param_grid.keys()}
    best_score = best_row["mean_R2"]

    results_df.to_csv(save_dir/"xgb_grid_results.csv", index=False)
    with open(save_dir/"xgb_grid_best.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "best_r2": float(best_score)}, f, indent=2, ensure_ascii=False)

    print("\n✅ XGBoost Grid Search Done.")
    print(f"Best R2 = {best_score:.4f}")
    print(f"Best Params = {best_params}")

    # =======================================================
    #  Step 2: 用最佳参数进行十折交叉验证
    # =======================================================
    print("\n使用最优参数进行十折交叉验证评估...\n")

    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = []

    for fold, (train_idx, valid_idx) in enumerate(kf10.split(X, y), 1):
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            n_jobs=1,
            **best_params
        )
        Xtr, Xva = X.iloc[train_idx], X.iloc[valid_idx]
        ytr, yva = y[train_idx], y[valid_idx]
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xva)
        r2 = r2_score(yva, y_pred)
        rmse = np.sqrt(mean_squared_error(yva, y_pred))
        mae = mean_absolute_error(yva, y_pred)
        metrics.append({"fold": fold, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"[Fold {fold:02d}] R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(save_dir/"xgb_cv10_metrics.csv", index=False)

    summary = {
        "R2_mean": df_metrics["R2"].mean(),
        "R2_std": df_metrics["R2"].std(ddof=1),
        "RMSE_mean": df_metrics["RMSE"].mean(),
        "MAE_mean": df_metrics["MAE"].mean(),
    }
    with open(save_dir/"xgb_cv10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n十折交叉验证完成")
    print(df_metrics)
    print(f"\n平均 R2 = {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}")

if __name__ == "__main__":
    main()
