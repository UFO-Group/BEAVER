#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ExtraTrees 预测脚本
- 自动识别 CSV 编码
- 自动识别 Morgan / 降维 / 复合特征列
- 优先使用模型训练时记录的 feature_names_in_ 顺序
- 自动对齐输入维度
- 将预测结果回并到原始 SMILES 表中

兼容你的训练脚本 LGBM_Catboost_Extratree.py：
- 默认读取 model.pkl
- 若传入的是总输出目录，也会自动尝试 {model_dir}/extratrees/model.pkl
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


DEFAULT_FEATURE_PATTERNS = (
    "poly_fp_", "add_fp_", "fp_dr_",
    "fp_", "morgan_", "ecfp_", "desc_", "frag_", "idx_", "pair_",
    "num_", "has_", "poly_fg_", "poly_phys_"
)


def smart_read_csv(path: Path, encoding: str = None) -> pd.DataFrame:
    if encoding is not None:
        print(f"[INFO] 使用指定编码读取 {path}: {encoding}")
        return pd.read_csv(path, encoding=encoding)

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


def resolve_model_path(model_dir: Path, model_name: str = "model.pkl") -> Path:
    candidates = [
        model_dir / model_name,
        model_dir / "extratrees" / model_name,
        model_dir / "ExtraTrees" / model_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "未找到 ExtraTrees 模型文件。已尝试:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
    )


def pick_feature_columns(df_feat: pd.DataFrame, model) -> list:
    # 1) 优先使用模型训练时记录的特征名与顺序
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        model_feature_names = list(model_feature_names)
        missing = [c for c in model_feature_names if c not in df_feat.columns]
        if not missing:
            print(f"[INFO] 使用模型内记录的特征顺序，共 {len(model_feature_names)} 列")
            return model_feature_names
        print(f"[WARN] 模型记录的特征名有 {len(missing)} 列在输入文件中缺失，将退回到前缀自动识别")

    # 2) 否则按前缀自动识别
    feature_cols = [
        c for c in df_feat.columns
        if any(c.startswith(p) for p in DEFAULT_FEATURE_PATTERNS)
        and np.issubdtype(df_feat[c].dtype, np.number)
    ]
    if not feature_cols:
        raise ValueError(
            "未检测到特征列。请检查列名前缀是否包含：\n"
            "poly_fp_/add_fp_/fp_dr_/fp_/morgan_/ecfp_/desc_/frag_/idx_/pair_/\n"
            "num_/has_/poly_fg_/poly_phys_"
        )
    print(f"[INFO] 按前缀自动识别到 {len(feature_cols)} 个特征列")
    return feature_cols


def main():
    ap = argparse.ArgumentParser(description="ExtraTrees 模型预测并回并原始 SMILES")
    ap.add_argument("--in_csv", required=True, help="输入：已pool的特征文件")
    ap.add_argument("--source_csv", required=True, help="输入：原始SMILES文件（用于合并输出）")
    ap.add_argument("--out_csv", required=True, help="输出：原始SMILES + 预测结果")
    ap.add_argument("--model_dir", required=True, help="模型目录；可传总输出目录或 extratrees 子目录")
    ap.add_argument("--model_name", default="model.pkl", help="模型文件名，默认 model.pkl")
    ap.add_argument("--target_name", type=str, default="Prediction", help="预测列名")
    ap.add_argument("--id_col", type=str, default="row_index", help="匹配索引列（默认 row_index）")
    ap.add_argument("--feat_encoding", type=str, default=None, help="特征文件编码，可选")
    ap.add_argument("--source_encoding", type=str, default=None, help="原始SMILES文件编码，可选")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    src_csv = Path(args.source_csv)
    out_csv = Path(args.out_csv)
    model_dir = Path(args.model_dir)

    if not in_csv.exists():
        raise FileNotFoundError(f"输入特征文件不存在: {in_csv}")
    if not src_csv.exists():
        raise FileNotFoundError(f"原始SMILES文件不存在: {src_csv}")

    model_path = resolve_model_path(model_dir, args.model_name)
    print(f"[INFO] 使用模型文件: {model_path}")

    df_feat = smart_read_csv(in_csv, encoding=args.feat_encoding)
    df_src = smart_read_csv(src_csv, encoding=args.source_encoding)
    print(f"[INFO] 特征文件: {df_feat.shape}, SMILES文件: {df_src.shape}")

    if args.id_col not in df_feat.columns:
        df_feat[args.id_col] = np.arange(len(df_feat))
        print(f"[INFO] 特征文件缺少 {args.id_col}，已自动补充")
    if args.id_col not in df_src.columns:
        df_src[args.id_col] = np.arange(len(df_src))
        print(f"[INFO] 原始文件缺少 {args.id_col}，已自动补充")

    model = joblib.load(model_path)
    feature_cols = pick_feature_columns(df_feat, model)
    print(f"[INFO] 前10个特征列示例: {feature_cols[:10]}")

    X_df = df_feat[feature_cols].copy()
    X = X_df.astype(np.float32).values
    nfeat_model = getattr(model, "n_features_in_", None)

    if nfeat_model is not None and X.shape[1] != nfeat_model:
        diff = nfeat_model - X.shape[1]
        if diff > 0:
            X = np.hstack([X, np.zeros((X.shape[0], diff), dtype=np.float32)])
            print(f"[AUTO] 输入少了 {diff} 列，已补零 -> {X.shape[1]} 维")
        else:
            X = X[:, :nfeat_model]
            print(f"[AUTO] 输入多了 {-diff} 列，已截断 -> {X.shape[1]} 维")

    y_pred = model.predict(X)
    df_feat[args.target_name] = y_pred
    print(f"[OK] ExtraTrees 回归预测完成，共 {len(y_pred)} 条")

    merged = pd.merge(
        df_src,
        df_feat[[args.id_col, args.target_name]],
        on=args.id_col,
        how="left"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存预测结果 -> {out_csv}")
    print(f"[INFO] 使用特征维度: {X.shape[1]}  模型期望: {nfeat_model}")
    print(f"[INFO] 输出列: {list(merged.columns)}")


if __name__ == "__main__":
    main()
