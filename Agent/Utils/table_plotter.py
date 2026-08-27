# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # 🔥 强制使用非交互式后端，防止 Streamlit 线程报错

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 🔥 必须导入，用于3D绘图（即便未显式使用也别删）
import pandas as pd
import numpy as np
import os
import re
import platform
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hashlib  # 🔥 用于生成唯一数据指纹（去重）

# ==========================================
# 🔥 尝试导入 adjustText 用于自动修复标签重叠
# ==========================================
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
    print("[Plotter] ✅ adjustText library detected. Label placement will be optimized.")
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("[Plotter] ⚠️ adjustText library NOT found. Labels may overlap. Install with: pip install adjustText")

# ==========================================
# 🔧 字体配置 + 全局绘图风格（对齐第二个脚本）
# ==========================================
def configure_fonts():
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    elif system == "Darwin":
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

configure_fonts()

FS_TITLE = 24
FS_LABEL = 22
FS_TICK = 18
FS_ANNOT = 16
FS_LEGEND = 16

AXIS_LW = 2.5
GRID_COLOR = "#E6E6E6"
LEGEND_FACE = "white"

def style_spines(ax, lw=AXIS_LW):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(lw)
    ax.spines["bottom"].set_linewidth(lw)

def style_ticks(ax):
    ax.tick_params(axis='both', labelsize=FS_TICK, width=2.0, colors='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')

def style_grid(ax):
    ax.grid(True, which="both", linestyle='--', alpha=0.35, color=GRID_COLOR, zorder=0)

def style_legend(ax, loc='best', **kwargs):
    leg = ax.legend(
        loc=loc,
        frameon=True,
        framealpha=1.0,
        facecolor=LEGEND_FACE,
        edgecolor='none',
        fontsize=FS_LEGEND,
        **kwargs,
    )
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontweight('bold')
            txt.set_color('black')
    return leg

def style_3d_axes(ax):
    ax.tick_params(axis='x', labelsize=FS_TICK, width=2.0, colors='black')
    ax.tick_params(axis='y', labelsize=FS_TICK, width=2.0, colors='black')
    ax.tick_params(axis='z', labelsize=FS_TICK, width=2.0, colors='black')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor('black')
        except Exception:
            pass
    try:
        ax.xaxis.line.set_color('black')
        ax.yaxis.line.set_color('black')
        ax.zaxis.line.set_color('black')
        ax.xaxis.line.set_linewidth(AXIS_LW)
        ax.yaxis.line.set_linewidth(AXIS_LW)
        ax.zaxis.line.set_linewidth(AXIS_LW)
    except Exception:
        pass
    try:
        ax.xaxis._axinfo['grid'].update(color=GRID_COLOR, linestyle='--', linewidth=1.0)
        ax.yaxis._axinfo['grid'].update(color=GRID_COLOR, linestyle='--', linewidth=1.0)
        ax.zaxis._axinfo['grid'].update(color=GRID_COLOR, linestyle='--', linewidth=1.0)
    except Exception:
        pass
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels(), ax.get_zticklabels()):
        for label in ticks:
            label.set_fontweight('bold')
            label.set_color('black')

# ==========================================
# 🛠️ 辅助函数：健壮的 CSV 读取
# ==========================================
def try_read_csv(path, usecols=None):
    """
    尝试使用多种编码读取 CSV 文件。
    ✅ 关键修复：如果 usecols 列不匹配，这不是编码问题，直接停止返回 None。
    """
    encodings_to_try = [
        'utf-8-sig',
        'utf-8',
        'gb18030',
        'gbk',
        'gb2312',
        'cp936',
        'latin1'
    ]

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                usecols=usecols,
                low_memory=False,
                encoding=enc
            )
            return df

        except UnicodeDecodeError:
            # 这是编码问题，继续尝试其他编码
            continue

        except ValueError as e:
            # ✅ usecols 不匹配是结构/列名问题，不是编码问题，直接退出
            msg = str(e)
            if "Usecols do not match columns" in msg or "usecols do not match columns" in msg:
                print(f"[CSV Loader] ❌ usecols 不匹配（不是编码问题），停止尝试其他编码: {e}")
                return None
            print(f"[CSV Loader] ⚠️ ValueError({enc}): {e}")
            return None

        except Exception as e:
            print(f"[CSV Loader] ⚠️ 使用 {enc} 读取时遇到非编码错误: {e}")
            continue

    print(f"[CSV Loader] ❌ 无法读取文件，已尝试所有编码: {encodings_to_try}")
    return None


# ==========================================
# 🌍 全局字段映射
# ==========================================
COL_MAP = {
    "tensile_strength":    "Tensile Strength (MPa)",
    "elongation_at_break": "Elongation at Break (%)",
    "youngs_modulus":      "Young's Modulus (kPa)",
    "glass_transition":    "Glass Transition Temperature (°C)",
    "melting_temperature": "Melting Temperature (°C)"
}

LABEL_MAP = {
    "tensile_strength":    "Tensile Strength (MPa)",
    "elongation_at_break": "Elongation at Break (%)",
    "youngs_modulus":      "Young's Modulus (MPa)",  # 绘图时会自动转 MPa
    "glass_transition":    "Tg (°C)",
    "melting_temperature": "Tm (°C)"
}

# ==========================================
# 🧼 数值清洗：把 "1,234" / "12±3" / ">100" / "~50" 这类转成可比较的 float 或 str
# ==========================================
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _to_float_or_str(x):
    """
    尝试从任意输入中抽取一个可比较的 float：
    - 1234 / "1,234" -> 1234.0
    - "12±3" -> 12.0
    - ">100" -> 100.0
    - "~50" -> 50.0
    抽不到就返回清洗后的字符串。
    """
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return str(x)

    s = str(x).strip()
    if s == "":
        return None

    # 去逗号
    s2 = s.replace(",", "")
    m = _NUM_RE.search(s2)
    if m:
        try:
            return float(m.group())
        except Exception:
            return s2
    return s2


# ==========================================
# 🛠️ 辅助函数：改进的数据去重 (🔥 关键修复)
# ==========================================
def deduplicate_results(results: list, cols_to_check: list = None) -> list:
    """
    改进的去重逻辑：
    - 用 (name + 指定列值指纹) 作为唯一标识，避免同名不同材料被误去重。
    - ✅ 支持属性值为字符串（如 12±3, >100, 1,234），指纹更稳定
    """
    if not results:
        return []

    seen_hashes = set()
    unique_items = []

    if cols_to_check is None:
        cols_to_check = list(COL_MAP.values())

    for item in results:
        name = item.get("Polymer A Name") or "Unknown Material"
        name_str = str(name).strip()

        prop_fingerprint = ""
        for col in cols_to_check:
            val = _to_float_or_str(item.get(col))
            if val is None:
                prop_fingerprint += "|NA"
            else:
                if isinstance(val, float):
                    prop_fingerprint += f"|{val:.4f}"
                else:
                    prop_fingerprint += f"|{val}"

        full_id_str = name_str + prop_fingerprint
        item_hash = hashlib.md5(full_id_str.encode('utf-8')).hexdigest()

        if item_hash not in seen_hashes:
            seen_hashes.add(item_hash)
            unique_items.append(item)

    return unique_items


# ==============================================================================
# 📊 1. 2D Ashby 图（非 log 版本）
# ==============================================================================
def plot_ashby_chart(
    target_results: list,
    db_path: str,
    save_path: str = "./",
    x_axis: str = "youngs_modulus",
    y_axis: str = "tensile_strength",
    tag: str | None = None,
):
    """
    绘制 2D Ashby Plot（原始数值，不使用 log 坐标）：
    - 背景：全量数据库 (灰色)
    - 前景：候选结果 (橙红色星标)
    """
    x_col_raw = COL_MAP.get(x_axis)
    y_col_raw = COL_MAP.get(y_axis)

    if not x_col_raw or not y_col_raw:
        print(f"[Plotter] ❌ Invalid axis keys: {x_axis}, {y_axis}")
        return None

    target_results = deduplicate_results(target_results, cols_to_check=[x_col_raw, y_col_raw])

    if not os.path.exists(db_path):
        print(f"[Plotter] ❌ Database not found: {db_path}")
        return None

    df_bg = try_read_csv(db_path, usecols=[x_col_raw, y_col_raw])
    if df_bg is None:
        print(f"[Plotter] ❌ Failed to read DB: {db_path}")
        return None

    # 背景数据清洗
    bg_x = pd.to_numeric(df_bg[x_col_raw], errors='coerce')
    bg_y = pd.to_numeric(df_bg[y_col_raw], errors='coerce')

    # 单位换算：Young's Modulus 从 kPa -> MPa
    if x_axis == "youngs_modulus":
        bg_x = bg_x / 1000.0
    if y_axis == "youngs_modulus":
        bg_y = bg_y / 1000.0

    # 非 log 模式：只去掉 NaN，不按 >0 过滤
    mask = bg_x.notna() & bg_y.notna()
    bg_x = bg_x[mask]
    bg_y = bg_y[mask]

    fg_x, fg_y, fg_labels = [], [], []
    for item in target_results:
        try:
            val_x = _to_float_or_str(item.get(x_col_raw, None))
            val_y = _to_float_or_str(item.get(y_col_raw, None))

            if val_x is None or val_y is None:
                continue

            if not isinstance(val_x, float):
                val_x = float(val_x)
            if not isinstance(val_y, float):
                val_y = float(val_y)

            if x_axis == "youngs_modulus":
                val_x /= 1000.0
            if y_axis == "youngs_modulus":
                val_y /= 1000.0

            if np.isfinite(val_x) and np.isfinite(val_y):
                fg_x.append(val_x)
                fg_y.append(val_y)
                name = str(item.get("Polymer A Name", "Item"))
                if len(name) > 15:
                    name = name[:12] + "..."
                fg_labels.append(name)
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    try:
        ax.scatter(
            bg_x, bg_y,
            c='#D3D3D3', s=40, alpha=0.5,
            label='Database (All Materials)',
            edgecolors='none', zorder=1
        )

        texts = []
        if fg_x:
            ax.scatter(
                fg_x, fg_y,
                c='#FF4500', s=220, marker='*', alpha=1.0,
                label='Selected Candidates',
                edgecolors='black', linewidth=1.2, zorder=10
            )

            for i in range(min(len(fg_x), 15)):
                t = ax.text(
                    fg_x[i], fg_y[i], fg_labels[i],
                    fontsize=FS_ANNOT, color='#8B0000', fontweight='bold'
                )
                texts.append(t)

        # ✅ 不再自动切换到 log 坐标

        ax.set_xlabel(LABEL_MAP.get(x_axis, x_axis), fontsize=FS_LABEL, fontweight='bold')
        ax.set_ylabel(LABEL_MAP.get(y_axis, y_axis), fontsize=FS_LABEL, fontweight='bold')
        ax.set_title(
            f"Material Property Space: {LABEL_MAP.get(y_axis, y_axis)} vs {LABEL_MAP.get(x_axis, x_axis)}",
            fontsize=FS_TITLE, fontweight='bold', pad=15
        )

        style_ticks(ax)
        style_spines(ax)
        style_grid(ax)
        style_legend(ax, loc='best')

        if ADJUST_TEXT_AVAILABLE and fg_x and texts:
            try:
                adjust_text(
                    texts,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    expand_points=(1.2, 1.2),
                    force_text=(0.2, 0.5)
                )
            except Exception as e:
                print(f"[Plotter] ⚠️ adjust_text warning: {e}")

        if save_path:
            save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        filename = f"Ashby_{y_axis}_vs_{x_axis}{suffix}.png"
        full_path = os.path.join(save_path, filename)

        plt.tight_layout()
        plt.savefig(full_path, bbox_inches='tight', facecolor='white')
        print(f"[Plotter] ✅ Chart saved: {full_path}")
        return full_path

    except Exception as e:
        print(f"[Plotter] ❌ Save failed: {e}")
        return None

    finally:
        plt.close(fig)


# ==============================================================================
# 🧊 3. 3D Ashby 图（非 log 版本）
# ==============================================================================
def plot_ashby_3d_chart(
    target_results: list,
    db_path: str,
    save_path: str = "./",
    x_axis: str = "youngs_modulus",
    y_axis: str = "tensile_strength",
    z_axis: str = "elongation_at_break",
    tag: str | None = None,
):
    """
    绘制 3D Ashby Plot（原始数值，不使用 log10）：
    - 背景：全量数据库原始值云团
    - 前景：候选点原始值星标
    """
    x_col = COL_MAP.get(x_axis)
    y_col = COL_MAP.get(y_axis)
    z_col = COL_MAP.get(z_axis)

    if not (x_col and y_col and z_col):
        print(f"[Plotter] ❌ Invalid 3D axis keys")
        return None

    target_results = deduplicate_results(target_results, cols_to_check=[x_col, y_col, z_col])

    if not os.path.exists(db_path):
        print(f"[Plotter] ❌ Database not found: {db_path}")
        return None

    df_bg = try_read_csv(db_path, usecols=[x_col, y_col, z_col])
    if df_bg is None:
        return None

    bg_x = pd.to_numeric(df_bg[x_col], errors='coerce')
    bg_y = pd.to_numeric(df_bg[y_col], errors='coerce')
    bg_z = pd.to_numeric(df_bg[z_col], errors='coerce')

    # 单位换算：Young's Modulus 从 kPa -> MPa
    if x_axis == "youngs_modulus":
        bg_x = bg_x / 1000.0
    if y_axis == "youngs_modulus":
        bg_y = bg_y / 1000.0
    if z_axis == "youngs_modulus":
        bg_z = bg_z / 1000.0

    # 非 log 模式：只去掉 NaN，不按 >0 过滤
    mask = bg_x.notna() & bg_y.notna() & bg_z.notna()
    bg_x, bg_y, bg_z = bg_x[mask], bg_y[mask], bg_z[mask]

    if len(bg_x) == 0:
        print("[Plotter] ⚠️ No valid background points for 3D plot.")
        return None

    fg_x, fg_y, fg_z, fg_labels = [], [], [], []
    for item in target_results:
        try:
            vx = _to_float_or_str(item.get(x_col, None))
            vy = _to_float_or_str(item.get(y_col, None))
            vz = _to_float_or_str(item.get(z_col, None))

            if vx is None or vy is None or vz is None:
                continue

            if not isinstance(vx, float):
                vx = float(vx)
            if not isinstance(vy, float):
                vy = float(vy)
            if not isinstance(vz, float):
                vz = float(vz)

            if x_axis == "youngs_modulus":
                vx /= 1000.0
            if y_axis == "youngs_modulus":
                vy /= 1000.0
            if z_axis == "youngs_modulus":
                vz /= 1000.0

            if np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz):
                fg_x.append(vx)
                fg_y.append(vy)
                fg_z.append(vz)
                name = str(item.get("Polymer A Name", "Item"))
                if len(name) > 12:
                    name = name[:10] + "..."
                fg_labels.append(name)
        except Exception:
            continue

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    try:
        ax.scatter(
            bg_x, bg_y, bg_z,
            c='#E0E0E0', s=15, alpha=0.12,
            edgecolors='none', label='Universe', zorder=1
        )

        if fg_x:
            fg_x = np.array(fg_x)
            fg_y = np.array(fg_y)
            fg_z = np.array(fg_z)

            ax.scatter(
                fg_x, fg_y, fg_z,
                c='#FF4500', s=180, marker='*', alpha=1.0,
                edgecolors='white', linewidth=0.5,
                label='Candidates', zorder=10, depthshade=False
            )

            for i in range(min(len(fg_x), 15)):
                dx = 0.02 * (np.nanmax(bg_x) - np.nanmin(bg_x) + 1e-12)
                dz = 0.02 * (np.nanmax(bg_z) - np.nanmin(bg_z) + 1e-12)
                ax.text(
                    fg_x[i] + dx,
                    fg_y[i],
                    fg_z[i] + dz,
                    fg_labels[i],
                    fontsize=FS_ANNOT,
                    color='#8B0000',
                    fontweight='bold',
                    zorder=20
                )

        ax.set_xlabel(LABEL_MAP.get(x_axis, x_axis), labelpad=14, fontsize=FS_LABEL, fontweight='bold')
        ax.set_ylabel(LABEL_MAP.get(y_axis, y_axis), labelpad=14, fontsize=FS_LABEL, fontweight='bold')
        ax.set_zlabel(LABEL_MAP.get(z_axis, z_axis), labelpad=14, fontsize=FS_LABEL, fontweight='bold')
        ax.set_title(
            f"3D Material Space: {LABEL_MAP.get(x_axis, x_axis)} / {LABEL_MAP.get(y_axis, y_axis)} / {LABEL_MAP.get(z_axis, z_axis)}",
            pad=20, fontsize=FS_TITLE, fontweight='bold'
        )

        ax.view_init(elev=30, azim=-60)
        style_3d_axes(ax)
        style_legend(ax, loc='upper left')

        if save_path:
            save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        filename = f"Ashby_3D_{x_axis}_{y_axis}_{z_axis}{suffix}.png"
        full_path = os.path.join(save_path, filename)

        plt.tight_layout()
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        print(f"[Plotter] ✅ 3D Chart saved: {full_path}")
        return full_path

    except Exception as e:
        print(f"[Plotter] ❌ Save 3D failed: {e}")
        return None

    finally:
        plt.close(fig)


# ==============================================================================
# 🧬 4. PCA 降维图 (🔥 修复标签重叠 + 支持 tag 文件名避免覆盖)
# ==============================================================================
def plot_pca_chart(
    target_results: list,
    db_path: str,
    save_path: str = "./",
    tag: str | None = None,
):
    """
    绘制 PCA 2D 投影：
    - 背景：全量材料空间（灰色）
    - 前景：候选点（紫色菱形）
    """
    cols_to_use = list(COL_MAP.values())

    # ✅ 去重：用所有 5D 属性
    target_results = deduplicate_results(target_results, cols_to_check=cols_to_use)

    if not os.path.exists(db_path):
        print(f"[Plotter] ❌ Database not found: {db_path}")
        return None

    df_bg = try_read_csv(db_path, usecols=cols_to_use)
    if df_bg is None:
        print("[Plotter] ❌ Failed to read CSV for PCA.")
        return None

    # 数据清洗：数值化 + dropna
    for col in cols_to_use:
        df_bg[col] = pd.to_numeric(df_bg[col], errors='coerce')
    df_bg.dropna(inplace=True)

    # ✅ 单位统一：Young's Modulus (kPa) -> MPa
    ym_col = COL_MAP["youngs_modulus"]
    if ym_col in df_bg.columns:
        df_bg[ym_col] = df_bg[ym_col] / 1000.0

    if len(df_bg) < 10:
        print("[Plotter] ⚠️ Not enough data points for PCA.")
        return None

    fg_matrix, fg_names = [], []
    for item in target_results:
        try:
            vec = []
            valid = True
            for col in cols_to_use:
                v = _to_float_or_str(item.get(col, np.nan))
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    valid = False
                    break
                if not isinstance(v, float):
                    v = float(v)
                vec.append(v)

            # ✅ 同样对前景 Young's Modulus 做单位统一
            # COL_MAP["youngs_modulus"] = "Young's Modulus (kPa)" 位于 vec 中
            if valid:
                ym_idx = cols_to_use.index(ym_col)
                vec[ym_idx] = vec[ym_idx] / 1000.0

                fg_matrix.append(vec)
                name = str(item.get("Polymer A Name", "Item"))
                if len(name) > 15:
                    name = name[:12] + "..."
                fg_names.append(name)
        except Exception:
            pass

    if not fg_matrix:
        print("[Plotter] ⚠️ No valid foreground data for PCA (missing properties).")
        return None

    bg_matrix = df_bg[cols_to_use].values
    combined_matrix = np.vstack([bg_matrix, np.array(fg_matrix)])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_matrix)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    n_bg = len(bg_matrix)
    pca_bg = pca_result[:n_bg]
    pca_fg = pca_result[n_bg:]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    try:
        ax.scatter(
            pca_bg[:, 0], pca_bg[:, 1],
            c='#E0E0E0', s=30, alpha=0.4,
            label='Material Universe (All Data)',
            edgecolors='none', zorder=1
        )

        ax.scatter(
            pca_fg[:, 0], pca_fg[:, 1],
            c='#9400D3', s=170, marker='D', alpha=0.9,
            label='Your Candidates',
            edgecolors='white', linewidth=1.0, zorder=10
        )

        texts = []
        for i in range(min(len(pca_fg), 15)):
            t = ax.text(
                pca_fg[i, 0], pca_fg[i, 1], fg_names[i],
                fontsize=FS_ANNOT, fontweight='bold', color='#4B0082'
            )
            texts.append(t)

        explained_var = pca.explained_variance_ratio_
        total_var = float(np.sum(explained_var) * 100)

        ax.set_xlabel(f"Principal Component 1 ({explained_var[0]:.1%} variance)", fontsize=FS_LABEL, fontweight='bold')
        ax.set_ylabel(f"Principal Component 2 ({explained_var[1]:.1%} variance)", fontsize=FS_LABEL, fontweight='bold')
        ax.set_title(
            f"PCA Map: 5D Property Space -> 2D (Info Retained: {total_var:.1f}%)",
            fontsize=FS_TITLE, fontweight='bold', pad=15
        )

        style_ticks(ax)
        style_spines(ax)
        style_grid(ax)
        style_legend(ax, loc='best')

        if ADJUST_TEXT_AVAILABLE and texts:
            try:
                adjust_text(
                    texts,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6),
                    expand_points=(1.5, 1.5)
                )
            except Exception as e:
                print(f"[Plotter] ⚠️ adjust_text warning: {e}")

        if save_path:
            save_path = os.path.abspath(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        filename = f"PCA_Material_Space{suffix}.png"
        full_path = os.path.join(save_path, filename)

        plt.tight_layout()
        plt.savefig(full_path, bbox_inches='tight', facecolor='white')
        print(f"[Plotter] ✅ PCA Chart saved: {full_path}")
        return full_path

    except Exception as e:
        print(f"[Plotter] ❌ Save PCA failed: {e}")
        return None

    finally:
        plt.close(fig)
