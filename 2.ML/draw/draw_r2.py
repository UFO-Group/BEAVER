#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制散点分布图（Train + test）
风格完全匹配直方图脚本
输出：
  1. scatter_log10.png （主图）
  2. scatter_linear.png
  3. scatter_test_only.png
  4. metrics_summary.json
"""

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MultipleLocator, MaxNLocator, ScalarFormatter
# ===== 全局统一参数（与直方图一致） =====
AX_SPINE_LW = 1.5   # 坐标轴外框线宽
TICK_W      = 1.5   # 刻度线粗细
TICK_LEN    = 6     # 刻度线长度
GRID_LW     = 1.0   # 网格线宽
MARKER_EDGE = 1.5   # 散点边框线宽
FIGSIZE     = (6, 6)
DPI         = 1000


# ===== 全局字体设为 Arial =====
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams['axes.formatter.useoffset'] = False
# ===== 统一字体与样式 =====
plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "axes.linewidth": AX_SPINE_LW,
    "axes.unicode_minus": False,
    "axes.grid": False,
})

# ===== metrics 函数 =====
def find_cols(df):
    for yt, yp in [("y_true", "y_pred"), ("target", "pred"), ("y", "yhat")]:
        if yt in df.columns and yp in df.columns:
            return yt, yp
    raise KeyError("Cannot find label/pred columns.")

def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def r2(a, b):
    den = np.sum((a - a.mean()) ** 2)
    return float(1.0 - np.sum((a - b) ** 2) / den) if den > 0 else float("nan")

def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    yt, yp = find_cols(df)
    y, yhat = df[yt].to_numpy(), df[yp].to_numpy()
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]

# ===== 绘图函数 =====
def scatter_train_test(y_tr, yhat_tr, y_va, yhat_va, out_png, log10=False):
    if log10:
        mask_tr = (y_tr > 0) & (yhat_tr > 0)
        mask_va = (y_va > 0) & (yhat_va > 0)
        y_tr, yhat_tr = np.log10(y_tr[mask_tr]), np.log10(yhat_tr[mask_tr])
        y_va, yhat_va = np.log10(y_va[mask_va]), np.log10(yhat_va[mask_va])

    r2_tr = r2(y_tr, yhat_tr)
    r2_va = r2(y_va, yhat_va)
    mae_tr = mae(y_tr, yhat_tr)
    mae_va = mae(y_va, yhat_va)
    rmse_tr = rmse(y_tr, yhat_tr)
    rmse_va = rmse(y_va, yhat_va)

    # ===== 绘图（风格完全匹配直方图） =====
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_box_aspect(1)  # 坐标区固定为正方形

    TRAIN_COLOR = (163/255, 197/255, 225/255)
    TEST_COLOR  = (183/255, 166/255, 208/255)

    plt.scatter(y_tr, yhat_tr, s=60, marker='o',
                facecolors=TRAIN_COLOR, edgecolors='black',
                linewidths=0.4, alpha=0.9, label="Train")
    plt.scatter(y_va, yhat_va, s=80, marker='^',
                facecolors=TEST_COLOR, edgecolors='black',
                linewidths=0.4, alpha=0.9, label="Test")
                

    lo = min(y_tr.min(), yhat_tr.min(), y_va.min(), yhat_va.min())
    hi = max(y_tr.max(), yhat_tr.max(), y_va.max(), yhat_va.max())
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1.2)


    
    if log10:
        ax.set_xlabel(r"True $\log_{10}$", fontsize=20)
        ax.set_ylabel(r"Pred. $\log_{10}$", fontsize=20)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))   # 或 1
        ax.yaxis.set_major_locator(MultipleLocator(0.5))   # 或 1
    else:
        ax.set_xlabel("True (%)", fontsize=20)
        ax.set_ylabel("Pred. (%)", fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    
    ax.set_title(r"$\log_{10}$ (Elongation at Break (%))")
    ax.legend(loc="upper left")


    # 坐标轴与刻度统一
    for s in ax.spines.values():
        s.set_linewidth(AX_SPINE_LW)
    ax.tick_params(axis="both", labelsize=26, width=TICK_W, length=TICK_LEN)

    # 添加性能指标
    txt = (
        f"$R^2_{{train}}$: {r2_tr:.2f}\n"
        f"$R^2_{{test}}$: {r2_va:.2f}\n"
        f"$\\mathrm{{MAE}}_{{train}}$: {mae_tr:.2f}\n"
        f"$\\mathrm{{MAE}}_{{test}}$: {mae_va:.2f}\n"
        f"$\\mathrm{{RMSE}}_{{train}}$: {rmse_tr:.2f}\n"
        f"$\\mathrm{{RMSE}}_{{test}}$: {rmse_va:.2f}"
    )
    ax.text(
        0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
        fontsize=16,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.3'),
        linespacing=1.1
    )

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return {"R2_train": r2_tr, "R2_test": r2_va,
            "MAE_train": mae_tr, "MAE_test": mae_va,
            "RMSE_train": rmse_tr, "RMSE_test": rmse_va}

# ===== 主函数 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--outdir", default="scatter_plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    y_tr, yhat_tr = load_xy(args.train_csv)
    y_va, yhat_va = load_xy(args.test_csv)

    # Linear 版本（保持log10轴标）
    met_lin = scatter_train_test(y_tr, yhat_tr, y_va, yhat_va,
                                  os.path.join(args.outdir, "scatter_linear.png"),
                                  log10=False)
    print("[OK] scatter_linear.png saved")

    # Log10版本
    y_tr, yhat_tr = load_xy(args.train_csv)
    y_va, yhat_va = load_xy(args.test_csv)
    met_log = scatter_train_test(y_tr, yhat_tr, y_va, yhat_va,
                                  os.path.join(args.outdir, "scatter_log10.png"),
                                  log10=True)
    print("[OK] scatter_log10.png saved")

    # test-only 图（匹配风格）
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True, dpi=DPI)
    mask_va = (y_va > 0) & (yhat_va > 0)
    y_va_log, yhat_va_log = np.log10(y_va[mask_va]), np.log10(yhat_va[mask_va])
    ax.scatter(
        y_va_log, yhat_va_log, s=80, facecolors='none',
        edgecolors='blue', linewidths=MARKER_EDGE, alpha=0.9
    )
    lo, hi = min(y_va_log.min(), yhat_va_log.min()), max(y_va_log.max(), yhat_va_log.max())
    ax.plot([lo, hi], [lo, hi], '--', color='red', linewidth=1.2)
    ax.set_xlabel(r"True $\log_{10}$", fontsize=20)
    ax.set_ylabel(r"Pred. $\log_{10}$", fontsize=20)
    ax.xaxis.set_major_locator(MultipleLocator(2)) 
    ax.yaxis.set_major_locator(MultipleLocator(2)) 
    ax.set_title("Tensile Strength (log10)")
    for s in ax.spines.values():
        s.set_linewidth(AX_SPINE_LW)
    ax.tick_params(axis="both", labelsize=20, width=TICK_W, length=TICK_LEN)
    plt.savefig(os.path.join(args.outdir, "scatter_test_only.png"), dpi=DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print("[OK] scatter_test_only.png saved")

    # 保存 metrics
    metrics = {"linear": met_lin, "log10": met_log}
    with open(os.path.join(args.outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[OK] metrics_summary.json saved")

if __name__ == "__main__":
    main()
