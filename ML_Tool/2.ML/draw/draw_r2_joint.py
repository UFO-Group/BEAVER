#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制带贴合式边缘 KDE 分布的 Train/Test R2 散点图

输出：
  1. scatter_linear_kde_attached.png
  2. scatter_log10_kde_attached.png
  3. scatter_test_only.png
  4. metrics_summary.json

特点：
- 中间主图：True vs Pred 散点（Train + Test）
- 上方边缘图：True 的 train/test 平滑密度曲线（面积归一化）
- 右侧边缘图：Pred 的 train/test 平滑密度曲线（面积归一化）
- 边缘分布与主图直接贴合，共享主图坐标范围
- 边缘分布不显示独立坐标，只保留主图坐标
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===== 全局参数 =====
AX_SPINE_LW = 2.0   # 坐标轴外框线宽
TICK_W      = 2.0   # 刻度线粗细
TICK_LEN    = 8     # 刻度线长度
MARKER_EDGE = 0.4
DPI = 1000
FIGSIZE = (6.8, 6.8)

MAIN_LABEL_FS = 26
MAIN_TITLE_FS = 26
MAIN_TICK_FS = 26
LEGEND_FS = 18
TEXTBOX_FS = 18

TRAIN_COLOR = (186/255, 212/255, 232/255)
TEST_COLOR = (161/255, 95/255, 235/255)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams.update({
    "axes.labelsize": MAIN_LABEL_FS,
    "axes.titlesize": MAIN_TITLE_FS,
    "xtick.labelsize": MAIN_TICK_FS,
    "ytick.labelsize": MAIN_TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "axes.linewidth": AX_SPINE_LW,
    "axes.unicode_minus": False,
    "axes.grid": False,
})


# ===== 工具函数 =====
def find_cols(df):
    for yt, yp in [("y_true", "y_pred"), ("target", "pred"), ("y", "yhat")]:
        if yt in df.columns and yp in df.columns:
            return yt, yp
    raise KeyError("Cannot find label/pred columns. Supported pairs: (y_true,y_pred), (target,pred), (y,yhat)")


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(a, b):
    den = np.sum((a - a.mean()) ** 2)
    return float(1.0 - np.sum((a - b) ** 2) / den) if den > 0 else float("nan")


def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    yt, yp = find_cols(df)
    y = df[yt].to_numpy(dtype=float)
    yhat = df[yp].to_numpy(dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]


def get_title(base_title, log10=False):
    if log10:
        return rf"$\log_{{10}}$ ({base_title})"
    return base_title


def get_axis_labels(log10=False):
    if log10:
        return r"True $\log_{10}$", r"Pred. $\log_{10}$"
    return r"True $\log_{10} \sigma_{\mathrm{b}}/\mathrm{MPa}$", r"Pred. $\log_{10} \sigma_{\mathrm{b}}/\mathrm{MPa}$"
# Young's modulus
# r"True $\log_{10} E/\mathrm{kPa}$", r"Pred. $\log_{10} E/\mathrm{kPa}$"
# Elongation at break
# r"True $\log_{10} \varepsilon_{\mathrm{b}}/\%$", r"Pred. $\log_{10} \varepsilon_{\mathrm{b}}/\%$"
# Tensile strength
# r"True $\log_{10} \sigma_{\mathrm{b}}/\mathrm{MPa}$", r"Pred. $\log_{10} \sigma_{\mathrm{b}}/\mathrm{MPa}$"
# Tm
# r"True $T_{\mathrm{m}}/^\circ\mathrm{C}$", r"Pred. $T_{\mathrm{m}}/^\circ\mathrm{C}$"
# Tg
# r"True $T_{\mathrm{g}}/^\circ\mathrm{C}$", r"Pred. $T_{\mathrm{g}}/^\circ\mathrm{C}$"

def get_limits(*arrays, margin_ratio=0.03):
    lo = min(np.min(arr) for arr in arrays)
    hi = max(np.max(arr) for arr in arrays)
    if np.isclose(lo, hi):
        delta = 1.0 if np.isfinite(lo) else 1.0
        lo -= delta
        hi += delta
    else:
        pad = (hi - lo) * margin_ratio
        lo -= pad
        hi += pad
    return lo, hi


def style_spines(ax):
    for s in ax.spines.values():
        s.set_linewidth(AX_SPINE_LW)


def add_metric_box(ax, y_tr, yhat_tr, y_va, yhat_va):
    r2_tr = r2(y_tr, yhat_tr)
    r2_va = r2(y_va, yhat_va)
    mae_tr = mae(y_tr, yhat_tr)
    mae_va = mae(y_va, yhat_va)
    rmse_tr = rmse(y_tr, yhat_tr)
    rmse_va = rmse(y_va, yhat_va)

    txt = (
        f"$R^2_{{train}}$: {r2_tr:.2f}\n"
        f"$R^2_{{test}}$: {r2_va:.2f}\n"
        f"$\\mathrm{{MAE}}_{{train}}$: {mae_tr:.2f}\n"
        f"$\\mathrm{{MAE}}_{{test}}$: {mae_va:.2f}\n"
        f"$\\mathrm{{RMSE}}_{{train}}$: {rmse_tr:.2f}\n"
        f"$\\mathrm{{RMSE}}_{{test}}$: {rmse_va:.2f}"
    )
    ax.text(
        0.98, 0.02, txt,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=TEXTBOX_FS,
        linespacing=1.08,
        zorder=10,
    )
    return {
        "R2_train": r2_tr,
        "R2_test": r2_va,
        "MAE_train": mae_tr,
        "MAE_test": mae_va,
        "RMSE_train": rmse_tr,
        "RMSE_test": rmse_va,
    }


def gaussian_kde_np(data, grid, bandwidth_scale=1.0):
    """纯 numpy KDE，输出为面积归一化密度。"""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = data.size
    if n == 0:
        return np.zeros_like(grid)
    if n == 1:
        bw = 0.1 if np.isfinite(data[0]) else 1.0
    else:
        std = np.std(data, ddof=1)
        if (not np.isfinite(std)) or std <= 1e-12:
            std = max(np.abs(np.mean(data)) * 0.05, 0.1)
        bw = 1.06 * std * (n ** (-1.0 / 5.0))
    bw *= bandwidth_scale
    bw = max(bw, 1e-3)

    z = (grid[:, None] - data[None, :]) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))

    area = np.trapz(dens, grid)
    if area > 0:
        dens = dens / area
    return dens


def style_marginal_top(ax):
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AX_SPINE_LW)


def style_marginal_right(ax):
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_density_top(ax, x_data, grid, color, alpha_fill=0.28, lw=2.0, bandwidth_scale=1.0):
    dens = gaussian_kde_np(x_data, grid, bandwidth_scale=bandwidth_scale)
    ax.fill_between(grid, 0.0, dens, color=color, alpha=alpha_fill, linewidth=0)
    ax.plot(grid, dens, color=color, linewidth=lw)
    return dens


def plot_density_right(ax, y_data, grid, color, alpha_fill=0.28, lw=2.0, bandwidth_scale=1.0):
    dens = gaussian_kde_np(y_data, grid, bandwidth_scale=bandwidth_scale)
    ax.fill_betweenx(grid, 0.0, dens, color=color, alpha=alpha_fill, linewidth=0)
    ax.plot(dens, grid, color=color, linewidth=lw)
    return dens


def make_kde_scatter_attached(
    y_tr, yhat_tr, y_va, yhat_va, out_png, base_title, log10=False,
    linear_tick_nbins=6, log_major_step=0.5, bandwidth_scale=1.0,
    marginal_size="20%", marginal_pad=0.0
):
    if log10:
        mask_tr = (y_tr > 0) & (yhat_tr > 0)
        mask_va = (y_va > 0) & (yhat_va > 0)
        y_tr = np.log10(y_tr[mask_tr])
        yhat_tr = np.log10(yhat_tr[mask_tr])
        y_va = np.log10(y_va[mask_va])
        yhat_va = np.log10(yhat_va[mask_va])

    fig, ax_main = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size=marginal_size, pad=marginal_pad, sharex=ax_main)
    ax_right = divider.append_axes("right", size=marginal_size, pad=marginal_pad, sharey=ax_main)

    lo, hi = get_limits(y_tr, yhat_tr, y_va, yhat_va, margin_ratio=0.03)
    grid = np.linspace(lo, hi, 500)

    # 主图
    ax_main.scatter(
        y_tr, yhat_tr,
        s=60, marker="o",
        facecolors=TRAIN_COLOR, edgecolors="black",
        linewidths=MARKER_EDGE, alpha=0.90, label="Train",
        zorder=3,
    )
    ax_main.scatter(
        y_va, yhat_va,
        s=80, marker="^",
        facecolors=TEST_COLOR, edgecolors="black",
        linewidths=MARKER_EDGE, alpha=0.90, label="Test",
        zorder=3,
    )
    ax_main.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_box_aspect(1)

    # 边缘 KDE：面积归一化
    d1 = plot_density_top(ax_top, y_tr, grid, TRAIN_COLOR, bandwidth_scale=bandwidth_scale)
    d2 = plot_density_top(ax_top, y_va, grid, TEST_COLOR, bandwidth_scale=bandwidth_scale)
    ax_top.set_ylim(0, max(np.max(d1), np.max(d2)) * 1.10)

    d3 = plot_density_right(ax_right, yhat_tr, grid, TRAIN_COLOR, bandwidth_scale=bandwidth_scale)
    d4 = plot_density_right(ax_right, yhat_va, grid, TEST_COLOR, bandwidth_scale=bandwidth_scale)
    ax_right.set_xlim(0, max(np.max(d3), np.max(d4)) * 1.10)

    xlabel, ylabel = get_axis_labels(log10=log10)
    ax_main.set_xlabel(xlabel, fontsize=MAIN_LABEL_FS)
    ax_main.set_ylabel(ylabel, fontsize=MAIN_LABEL_FS)
    ax_main.set_title(get_title(base_title, log10=log10), fontsize=MAIN_TITLE_FS, pad=8)
    ax_main.legend(loc="upper left", frameon=False)

    if log10:
        ax_main.xaxis.set_major_locator(MultipleLocator(log_major_step))
        ax_main.yaxis.set_major_locator(MultipleLocator(log_major_step))
    else:
        ax_main.xaxis.set_major_locator(MaxNLocator(nbins=linear_tick_nbins))
        ax_main.yaxis.set_major_locator(MaxNLocator(nbins=linear_tick_nbins))
        ax_main.ticklabel_format(style="plain", axis="both", useOffset=False)

    ax_main.tick_params(axis="both", labelsize=MAIN_TICK_FS, width=TICK_W, length=TICK_LEN)

    style_spines(ax_main)
    style_marginal_top(ax_top)
    style_marginal_right(ax_right)

    metrics = add_metric_box(ax_main, y_tr, yhat_tr, y_va, yhat_va)

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return metrics


def make_test_only_plot(y_va, yhat_va, out_png, base_title, log_major_step=0.5):
    mask_va = (y_va > 0) & (yhat_va > 0)
    y_va_log = np.log10(y_va[mask_va])
    yhat_va_log = np.log10(yhat_va[mask_va])

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=DPI)
    ax.set_box_aspect(1)

    ax.scatter(
        y_va_log, yhat_va_log,
        s=80, marker="^",
        facecolors="none", edgecolors="blue",
        linewidths=1.2, alpha=0.90,
    )

    lo, hi = get_limits(y_va_log, yhat_va_log, margin_ratio=0.03)
    ax.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(r"True $\log_{10}$", fontsize=MAIN_LABEL_FS)
    ax.set_ylabel(r"Pred. $\log_{10}$", fontsize=MAIN_LABEL_FS)
    ax.set_title(rf"$\log_{{10}}$ ({base_title}) - Test", fontsize=MAIN_TITLE_FS)
    ax.xaxis.set_major_locator(MultipleLocator(log_major_step))
    ax.yaxis.set_major_locator(MultipleLocator(log_major_step))
    ax.tick_params(axis="both", labelsize=20, width=TICK_W, length=TICK_LEN)

    style_spines(ax)
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--outdir", default="scatter_plots_kde_attached")
    ap.add_argument("--title", default="Elongation at Break (%)",
                    help="图标题基名；log10 图会自动显示为 log10(title)")
    ap.add_argument("--linear_tick_nbins", type=int, default=6,
                    help="linear 主图主刻度数量")
    ap.add_argument("--log_major_step", type=float, default=0.5,
                    help="log10 主图主刻度步长")
    ap.add_argument("--bandwidth_scale", type=float, default=1.0,
                    help="KDE 带宽缩放；>1 更平滑，<1 更贴数据")
    ap.add_argument("--marginal_size", default="20%",
                    help="边缘 KDE 图相对主图尺寸，例如 18%% / 20%% / 24%%")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    y_tr, yhat_tr = load_xy(args.train_csv)
    y_va, yhat_va = load_xy(args.test_csv)

    met_lin = make_kde_scatter_attached(
        y_tr, yhat_tr, y_va, yhat_va,
        out_png=os.path.join(args.outdir, "scatter_linear_kde_attached.png"),
        base_title=args.title,
        log10=False,
        linear_tick_nbins=args.linear_tick_nbins,
        log_major_step=args.log_major_step,
        bandwidth_scale=args.bandwidth_scale,
        marginal_size=args.marginal_size,
        marginal_pad=0.0,
    )
    print("[OK] scatter_linear_kde_attached.png saved")

    y_tr, yhat_tr = load_xy(args.train_csv)
    y_va, yhat_va = load_xy(args.test_csv)
    met_log = make_kde_scatter_attached(
        y_tr, yhat_tr, y_va, yhat_va,
        out_png=os.path.join(args.outdir, "scatter_log10_kde_attached.png"),
        base_title=args.title,
        log10=True,
        linear_tick_nbins=args.linear_tick_nbins,
        log_major_step=args.log_major_step,
        bandwidth_scale=args.bandwidth_scale,
        marginal_size=args.marginal_size,
        marginal_pad=0.0,
    )
    print("[OK] scatter_log10_kde_attached.png saved")

    y_va, yhat_va = load_xy(args.test_csv)
    make_test_only_plot(
        y_va, yhat_va,
        out_png=os.path.join(args.outdir, "scatter_test_only.png"),
        base_title=args.title,
        log_major_step=args.log_major_step,
    )
    print("[OK] scatter_test_only.png saved")

    metrics = {"linear": met_lin, "log10": met_log}
    with open(os.path.join(args.outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[OK] metrics_summary.json saved")


if __name__ == "__main__":
    main()
