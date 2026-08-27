#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load


def read_csv_auto(path):
    tried = []
    for enc in ("utf-8-sig", "gb18030", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] 成功读取 {path}，编码: {enc}")
            return df
        except Exception as e:
            tried.append((enc, str(e)))
    msg = f"无法读取文件: {path}\n"
    for enc, err in tried:
        msg += f"  - {enc}: {err}\n"
    raise RuntimeError(msg)


def main():
    ap = argparse.ArgumentParser(description="RF模型预测并回并原始SMILES（支持回归与分类）")
    ap.add_argument("--in_csv", required=True, help="输入：已pool的Morgan或复合特征文件")
    ap.add_argument("--source_csv", required=True, help="输入：原始SMILES文件（用于合并输出）")
    ap.add_argument("--out_csv", required=True, help="输出：原始SMILES + 预测结果 (+ 分类概率)")
    ap.add_argument("--model_dir", required=True, help="模型目录（含 best_model.joblib）")
    ap.add_argument("--target_name", type=str, default="Prediction", help="预测列名")
    ap.add_argument("--id_col", type=str, default="row_index", help="匹配索引列（默认 row_index）")
    args = ap.parse_args()

    in_csv, src_csv, out_csv, model_dir = map(Path, [args.in_csv, args.source_csv, args.out_csv, args.model_dir])
    model_path = model_dir / "best_model.joblib"

    if not in_csv.exists():
        raise FileNotFoundError(f"输入特征文件不存在: {in_csv}")
    if not src_csv.exists():
        raise FileNotFoundError(f"原始SMILES文件不存在: {src_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # === 读取数据（自动识别编码） ===
    df_feat = read_csv_auto(in_csv)
    df_src = read_csv_auto(src_csv)
    print(f"[INFO] 特征文件: {df_feat.shape}, SMILES文件: {df_src.shape}")

    # === 检查并补 row_index ===
    if args.id_col not in df_feat.columns:
        df_feat[args.id_col] = np.arange(len(df_feat))
    if args.id_col not in df_src.columns:
        df_src[args.id_col] = np.arange(len(df_src))

    # === 加载模型 ===
    model_loaded = load(model_path)
    model = model_loaded
    if isinstance(model_loaded, dict):
        for k in ("model", "rf", "estimator", "pipe"):
            if k in model_loaded and hasattr(model_loaded[k], "predict"):
                model = model_loaded[k]
                break

    # === 自动识别特征列 ===
    FEATURE_PATTERNS = (
        "poly_fp_", "add_fp_", "fp_dr_",
        "fp_", "morgan_", "ecfp_", "desc_", "frag_", "idx_", "pair_"
    )
    feature_cols = [
        c for c in df_feat.columns
        if any(c.startswith(p) for p in FEATURE_PATTERNS)
        and np.issubdtype(df_feat[c].dtype, np.number)
    ]
    if not feature_cols:
        raise ValueError(
            "未检测到特征列。请检查列名前缀是否包含 "
            "poly_fp_/add_fp_/fp_dr_/fp_/desc_/frag_/idx_/pair_。"
        )

    X = df_feat[feature_cols].astype(np.float32).values
    nfeat_model = getattr(model, "n_features_in_", None)
    print(f"[INFO] 自动识别到 {len(feature_cols)} 个特征列，前10列示例: {feature_cols[:10]}")

    # === 自动维度对齐 ===
    if nfeat_model and X.shape[1] != nfeat_model:
        diff = nfeat_model - X.shape[1]
        if diff > 0:
            X = np.hstack([X, np.zeros((X.shape[0], diff), dtype=np.float32)])
            print(f"[AUTO] 补零列 {diff} -> 输入维度 {X.shape[1]}")
        else:
            X = X[:, :nfeat_model]
            print(f"[AUTO] 截断 {-diff} 列 -> 输入维度 {X.shape[1]}")

    # === 模型预测 ===
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        df_feat[args.target_name] = y_pred
        for i in range(y_prob.shape[1]):
            df_feat[f"{args.target_name}_prob_class{i}"] = y_prob[:, i]
        print(f"[OK] 分类预测完成，共 {len(y_pred)} 条，类别数={y_prob.shape[1]}")
    else:
        y_pred = model.predict(X)
        df_feat[args.target_name] = y_pred
        print(f"[OK] 回归预测完成，共 {len(y_pred)} 条。")

    merged = pd.merge(
        df_src,
        df_feat[[args.id_col] + [c for c in df_feat.columns if c.startswith(args.target_name)]],
        on=args.id_col,
        how="left"
    )
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] 已保存预测结果 -> {out_csv}")
    print(f"[INFO] 使用特征维度: {X.shape[1]}  模型期望: {nfeat_model}")


if __name__ == "__main__":
    main()