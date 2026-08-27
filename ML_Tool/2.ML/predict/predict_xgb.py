#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost / sklearn-bundle 预测脚本
- 读取 baseline_xgboost_mlp_svm.py 保存的 best_model.joblib
- 自动读取特征 CSV 与原始 SMILES CSV
- 严格按训练时保存的 features 对齐列顺序
- 缺失特征自动补 0，多余特征自动忽略
- 输出：原始 SMILES 列 + 预测结果列
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")


def read_csv_auto(path: Path) -> pd.DataFrame:
    tried = []
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] 成功读取 {path}，编码: {enc}")
            return df
        except Exception as e:
            tried.append((enc, str(e)))
    msg = [f"无法读取文件: {path}"]
    for enc, err in tried:
        msg.append(f"  - {enc}: {err}")
    raise RuntimeError("\n".join(msg))



def load_bundle(model_dir: Path, model_name: str) -> dict:
    candidates = []
    if model_name:
        candidates.append(model_dir / model_name)
    candidates.extend([
        model_dir / "best_model.joblib",
        model_dir / "xgb" / "best_model.joblib",
        model_dir / "xgboost" / "best_model.joblib",
    ])

    tried = []
    for path in candidates:
        if not path.exists():
            tried.append(f"不存在: {path}")
            continue
        obj = joblib.load(path)
        if isinstance(obj, dict) and "model" in obj and "features" in obj:
            print(f"[INFO] 成功加载模型包: {path}")
            return obj
        tried.append(f"不是有效 bundle: {path}")

    raise FileNotFoundError(
        "未找到可用的 best_model.joblib，或文件内容不是包含 'model' 和 'features' 的 bundle。\n"
        + "\n".join(tried)
    )



def main():
    ap = argparse.ArgumentParser(description="读取 best_model.joblib 并进行 XGBoost 预测")
    ap.add_argument("--in_csv", required=True, help="输入：特征 CSV")
    ap.add_argument("--source_csv", required=True, help="输入：原始 SMILES CSV，用于回并输出")
    ap.add_argument("--out_csv", required=True, help="输出：原始 SMILES + 预测结果")
    ap.add_argument("--model_dir", required=True, help="模型目录（内含 best_model.joblib）")
    ap.add_argument("--model_name", default="best_model.joblib", help="模型文件名，默认 best_model.joblib")
    ap.add_argument("--target_name", type=str, default="Prediction", help="预测列名")
    ap.add_argument("--id_col", type=str, default="row_index", help="匹配索引列，默认 row_index")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    source_csv = Path(args.source_csv)
    out_csv = Path(args.out_csv)
    model_dir = Path(args.model_dir)

    if not in_csv.exists():
        raise FileNotFoundError(f"输入特征文件不存在: {in_csv}")
    if not source_csv.exists():
        raise FileNotFoundError(f"原始 SMILES 文件不存在: {source_csv}")
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    df_feat = read_csv_auto(in_csv)
    df_src = read_csv_auto(source_csv)
    print(f"[INFO] 特征文件 shape: {df_feat.shape}, 原始文件 shape: {df_src.shape}")

    if args.id_col not in df_feat.columns:
        df_feat[args.id_col] = np.arange(len(df_feat))
    if args.id_col not in df_src.columns:
        df_src[args.id_col] = np.arange(len(df_src))

    bundle = load_bundle(model_dir, args.model_name)
    model = bundle["model"]
    feat_cols_ref = list(bundle["features"])
    target_in_bundle = bundle.get("target")
    print(f"[INFO] 训练时特征列数: {len(feat_cols_ref)}")
    if target_in_bundle is not None:
        print(f"[INFO] 训练目标列: {target_in_bundle}")

    # 只从当前特征表里取数值列，再按训练特征顺序补齐/截取
    df_num = df_feat.select_dtypes(include=[np.number]).copy()
    missing = [c for c in feat_cols_ref if c not in df_num.columns]
    extra = [c for c in df_num.columns if c not in feat_cols_ref and c != args.id_col]

    for c in missing:
        df_num[c] = 0.0

    X_df = df_num.reindex(columns=feat_cols_ref, fill_value=0.0).astype(np.float32)
    print(f"[INFO] 缺失特征补零数: {len(missing)}")
    print(f"[INFO] 额外数值列忽略数: {len(extra)}")
    if missing:
        print(f"[INFO] 缺失特征示例: {missing[:10]}")
    if extra:
        print(f"[INFO] 忽略特征示例: {extra[:10]}")

    y_pred = model.predict(X_df)
    df_feat[args.target_name] = y_pred
    print(f"[OK] 预测完成，共 {len(y_pred)} 条。")

    merged = pd.merge(
        df_src,
        df_feat[[args.id_col, args.target_name]],
        on=args.id_col,
        how="left",
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存预测结果 -> {out_csv}")
    print(f"[INFO] 输出列: {list(merged.columns)}")


if __name__ == "__main__":
    main()
