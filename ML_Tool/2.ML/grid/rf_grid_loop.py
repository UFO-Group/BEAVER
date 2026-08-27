#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, itertools, json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, GroupKFold


# ====== 工具函数 ======
def read_csv_robust(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def split_xy(df: pd.DataFrame, target: str, drop_cols: List[str]):
    exclude = set(drop_cols + [target])
    auto_exclude = {
        c for c in df.columns
        if any(key in str(c).lower() for key in ["index", "id", "recipe", "smiles", "unnamed"])
    }
    exclude |= auto_exclude

    # Only keep numeric feature columns
    cols = [
        c for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]

    if not cols:
        raise ValueError("No numeric feature columns detected. Please check your CSV file.")

    # ---- Logging ----
    print(f"[INFO] Detected {len(cols)} numeric feature columns.")
    print(f"[INFO] Example feature columns: {cols[:10]}")
    print(f"[INFO] Excluded columns: {sorted(exclude)}")

    X = df[cols].copy()
    y = df[target].values
    return X, y



def json_safe(obj):
    """确保对象可被json序列化（转换numpy类型为Python原生类型）"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [json_safe(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        return obj


def cast_to_number_if_possible(val):
    """尝试把字符串转为float/int，若失败则原样返回"""
    if isinstance(val, str):
        try:
            if val.isdigit():
                return int(val)
            f = float(val)
            return f
        except ValueError:
            return val
    return val


# ====== 主函数 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--id_cols", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=0, help="k-fold 数；>0 时启用交叉验证")
    ap.add_argument("--group_col", default="", help="分组列名（用于GroupKFold，可选）")
    ap.add_argument("--test_size", type=float, default=0.2, help="cv=0时使用的留出比例")
    ap.add_argument("--final_cv", type=int, default=10, help="最终再验证的折数")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_robust(args.in_csv)
    drop_cols = [c for c in args.id_cols.split(",") if c]
    X, y = split_xy(df, args.target, drop_cols=drop_cols)

    groups = None
    if args.group_col:
        if args.group_col not in df.columns:
            raise ValueError(f"group_col 不在数据列中: {args.group_col}")
        groups = df[args.group_col].values

    # ====== 网格参数 ======
    grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000],
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40],
        'min_samples_split': [8, 10, 12, 14, 16],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': [0.1, 0.2, 0.3, 0.5],
    }

    keys, values = zip(*grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    out_csv = save_dir / "rf_grid_results.csv"

    with open(out_csv, "w", encoding="utf-8") as fout:
        if args.cv and args.cv > 0:
            header = list(keys) + ["R2_train", "R2_cv_mean", "R2_cv_std"]
        else:
            header = list(keys) + ["R2_train", "R2_valid"]
        fout.write(",".join(header) + "\n")

        # ====== 交叉验证器 ======
        cv_splitter = None
        if args.cv and args.cv > 0:
            if groups is not None:
                cv_splitter = GroupKFold(n_splits=args.cv)
            else:
                cv_splitter = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

        # ====== 网格搜索 ======
        for i, params in enumerate(combos, 1):
            model = RandomForestRegressor(n_jobs=-1, random_state=args.seed, **params)
            model.fit(X, y)
            r2_train = r2_score(y, model.predict(X))

            if cv_splitter is not None:
                r2_list = []
                splits = cv_splitter.split(X, y, groups) if groups is not None else cv_splitter.split(X, y)
                for tr_idx, va_idx in splits:
                    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
                    ytr, yva = y[tr_idx], y[va_idx]
                    m = RandomForestRegressor(n_jobs=-1, random_state=args.seed, **params)
                    m.fit(Xtr, ytr)
                    r2_list.append(r2_score(yva, m.predict(Xva)))
                r2_cv_mean = float(np.mean(r2_list))
                r2_cv_std = float(np.std(r2_list))
                row = [params[k] for k in keys] + [r2_train, r2_cv_mean, r2_cv_std]
            else:
                Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
                m = RandomForestRegressor(n_jobs=-1, random_state=args.seed, **params)
                m.fit(Xtr, ytr)
                r2_valid = r2_score(yva, m.predict(Xva))
                row = [params[k] for k in keys] + [r2_train, r2_valid]

            fout.write(",".join(str(x) for x in row) + "\n")
            fout.flush()
            print(f"[{i}/{len(combos)}] {params} -> R2_train={r2_train:.4f} "
                  + (f"CV mean={r2_cv_mean:.4f} std={r2_cv_std:.4f}" if args.cv else f"R2_valid={r2_valid:.4f}"))

    print(f"\n[done] Grid search results saved to {out_csv}")

    # ====== 自动选最优参数并重新做10折CV ======
    df_res = pd.read_csv(out_csv)
    metric_col = "R2_cv_mean" if "R2_cv_mean" in df_res.columns else "R2_valid"
    best_row = df_res.loc[df_res[metric_col].idxmax()]
    best_params = {k: cast_to_number_if_possible(json_safe(best_row[k])) for k in keys}
    print(f"\n[Best Params] {best_params}")

    # === 强制转换为 int，防止 25.0 报错 ===
    for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
        if k in best_params and not isinstance(best_params[k], (str, type(None))):
            try:
                best_params[k] = int(float(best_params[k]))
            except Exception:
                pass

    # === 处理 max_features ===
    if isinstance(best_params.get("max_features"), str):
        try:
            val = float(best_params["max_features"])
            if 0 < val <= 1:
                best_params["max_features"] = val
        except ValueError:
            pass

    # 保存最佳参数
    best_param_path = save_dir / "best_params.json"
    best_param_path.write_text(json.dumps(best_params, indent=2, ensure_ascii=False))
    print(f"[saved] {best_param_path}")

    # ====== 用最佳参数执行10折CV ======
    print("\n[RUN] Final 10-fold CV with best params ...")
    kf = KFold(n_splits=args.final_cv, shuffle=True, random_state=args.seed)
    r2s, rmses, maes = [], [], []

    for i, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        m = RandomForestRegressor(n_jobs=-1, random_state=args.seed, **best_params)
        m.fit(Xtr, ytr)
        y_pred = m.predict(Xva)
        r2s.append(r2_score(yva, y_pred))
        rmses.append(np.sqrt(mean_squared_error(yva, y_pred)))
        maes.append(mean_absolute_error(yva, y_pred))
        print(f"[fold {i}] R2={r2s[-1]:.3f}, RMSE={rmses[-1]:.3f}, MAE={maes[-1]:.3f}")

    summary = {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
    }
    print("\n[Final CV Summary]")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    pd.DataFrame(summary, index=[0]).to_csv(save_dir / "final_cv_metrics.csv", index=False)
    print(f"[saved] final_cv_metrics.csv")


if __name__ == "__main__":
    main()
