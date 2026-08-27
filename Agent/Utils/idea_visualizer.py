import matplotlib
matplotlib.use('Agg')

import json
import os
import platform
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


# ==========================================
# Font configuration
# ==========================================
def configure_fonts():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "SimHei"]
    elif system == "Darwin":
        plt.rcParams["font.sans-serif"] = ["Arial", "PingFang SC", "Arial Unicode MS"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "WenQuanYi Micro Hei"]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


configure_fonts()


# ==========================================
# Global style constants
# ==========================================
FS_TITLE = 20
FS_LABEL = 17
FS_TICK = 14
FS_ANNOT = 12
FS_LEGEND = 12

ACCENT_COLOR = (120 / 255, 120 / 255, 120 / 255)
RADAR_COLORS = [
    (162 / 255, 204 / 255, 201 / 255),
    (234 / 255, 177 / 255, 200 / 255),
    (146 / 255, 190 / 255, 128 / 255),
]
GRID_COLOR = "#D9D9D9"

FINAL_RADAR_CATEGORIES = [
    "Feasibility",
    "Predictability",
    "Performance",
    "Innovation",
    "Chemical Validity",
]

FINAL_RADAR_LABELS = [
    "Feasibility",
    "Predictability",
    "Performance",
    "Innovation",
    "Chemical\nValidity",
]


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def style_spines(ax, lw=2.2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(lw)
    ax.spines["bottom"].set_linewidth(lw)


# ==========================================
# 1. Initial idea-ranking bar plot
# ==========================================
def plot_initial_ranking(ideas, save_path):
    if not ideas:
        return None

    os.makedirs(save_path, exist_ok=True)

    names = []
    scores = []

    for idx, i in enumerate(ideas):
        raw_name = str(i.get("idea_name", f"Idea {idx + 1}"))
        if len(raw_name) > 24:
            raw_name = raw_name[:21] + "..."
        names.append(raw_name)
        scores.append(safe_float(i.get("score_overall", 0), 0.0))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    dark_pink = (190 / 255, 100 / 255, 140 / 255)
    base_pink = (236 / 255, 183 / 255, 204 / 255)
    light_pink = (252 / 255, 238 / 255, 242 / 255)

    cmap = LinearSegmentedColormap.from_list(
        "custom_pink", [light_pink, base_pink, dark_pink]
    )
    norm = Normalize(vmin=min(scores) - 1.0, vmax=max(scores))
    bar_colors = [cmap(norm(s)) for s in scores]

    bars = ax.bar(
        names,
        scores,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.75,
        alpha=1.00,
        width=0.75,
        zorder=3,
    )

    if len(scores) >= 3:
        cutoff = sorted(scores, reverse=True)[2]
        ax.axhline(
            y=cutoff,
            color=ACCENT_COLOR,
            linestyle="--",
            linewidth=2.2,
            alpha=0.95,
            zorder=2,
        )
        ax.text(
            len(names) - 0.3,
            cutoff + 0.12,
            f"Selection Threshold ({cutoff:.2f})",
            color=ACCENT_COLOR,
            ha="right",
            va="bottom",
            fontsize=FS_ANNOT,
            fontweight="bold",
        )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.08,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=FS_ANNOT,
            fontweight="bold",
            color="black",
        )

    ax.set_ylabel("Overall Score (0-10)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"Idea Divergence Analysis: Top {len(ideas)} Candidates",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=12,
    )

    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)

    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_rotation_mode("anchor")
        label.set_ha("right")
        label.set_fontweight("bold")
        label.set_color("black")

    plt.subplots_adjust(bottom=0.24)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("black")

    ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR, zorder=0)
    style_spines(ax, lw=2.2)
    ax.set_ylim(0, max(11, max(scores) + 1.0))

    plt.tight_layout()
    target_file = os.path.join(save_path, "initial_screening_bar.png")
    plt.savefig(target_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return target_file


# ==========================================
# 2. Final report-level radar plot
# ==========================================
def _format_radar_title(title):
    title = str(title or "")
    if len(title) > 38:
        return title[:35] + "..."
    return title


def plot_final_radar(results, save_path):
    """
    Plot final report-level scores for the verified top candidates.

    Expected input format:
    results = [
        {
            "title": "Idea title",
            "scores_dict": {
                "Feasibility": 80,
                "Predictability": 75,
                "Performance": 82,
                "Innovation": 70,
                "Chemical Validity": 85,
            }
        },
        ...
    ]
    """
    if not results:
        return None

    os.makedirs(save_path, exist_ok=True)

    categories = FINAL_RADAR_CATEGORIES
    display_labels = FINAL_RADAR_LABELS

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(
        figsize=(8.4, 8.4),
        subplot_kw=dict(polar=True),
        dpi=300,
    )

    # Leave clean margins for labels and bottom legend.
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.14, right=0.86)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    # Axis labels: automatic 5-axis layout.
    ax.set_xticks(angles)
    ax.set_xticklabels(display_labels)
    ax.tick_params(axis="x", pad=24)

    for label in ax.get_xticklabels():
        label.set_fontsize(15)
        label.set_fontweight("bold")
        label.set_color("black")

    # Radial ticks: keep them small and unobtrusive.
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"])
    ax.set_rlabel_position(225)

    for label in ax.get_yticklabels():
        label.set_fontsize(11)
        label.set_fontweight("bold")
        label.set_color("#555555")

    # Grid and border.
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.9, alpha=0.85)
    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(1.6)

    for idx, res in enumerate(results):
        s_dict = res.get("scores_dict", {}) or {}
        values = [safe_float(s_dict.get(c, 0), 0.0) for c in categories]
        values_closed = values + values[:1]

        color = RADAR_COLORS[idx % len(RADAR_COLORS)]
        label_text = _format_radar_title(res.get("title", f"Idea {idx + 1}"))

        ax.plot(
            angles_closed,
            values_closed,
            linewidth=2.4,
            linestyle="-",
            label=label_text,
            color=color,
            alpha=0.96,
            zorder=4,
        )
        ax.fill(
            angles_closed,
            values_closed,
            color=color,
            alpha=0.13,
            zorder=2,
        )

    ax.set_title(
        "Final Candidate Evaluation",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=28,
        color="black",
    )

    # Bottom legend avoids clipping and keeps the radar area clean.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=1,
        frameon=False,
        fontsize=FS_LEGEND,
        handlelength=2.8,
    )

    target_file = os.path.join(save_path, "final_verification_radar.png")
    plt.savefig(target_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return target_file


# ==========================================
# 3. Standalone helper: regenerate radar without rerunning BEAVER
# ==========================================
def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_title_from_folder(folder_path):
    folder_name = Path(folder_path).name
    # Example: idea1_Aromatic_Aliphatic_Copolyester
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        return parts[1].replace("_", " ")
    return folder_name.replace("_", " ")


def load_final_results_from_run_dir(run_dir):
    """
    Load existing top-idea score sidecars from a finished Design run directory.
    This allows you to redraw final_verification_radar.png without rerunning the pipeline.

    Expected files:
        <run_dir>/idea1_*/idea1_scores.json
        <run_dir>/idea2_*/idea2_scores.json
        <run_dir>/idea3_*/idea3_scores.json
    """
    run_dir = os.path.normpath(str(run_dir))
    pattern = os.path.join(run_dir, "idea*_*/idea*_scores.json")
    score_files = sorted(glob.glob(pattern))

    results = []
    for score_file in score_files:
        try:
            data = _read_json(score_file)
        except Exception as e:
            print(f"[Skip] Failed to read {score_file}: {e}")
            continue

        idea_dir = os.path.dirname(score_file)
        idea_id = Path(score_file).stem.replace("_scores", "")
        scores_dict = data.get("scores_dict", {}) or {}

        # Old runs may not have Chemical Validity. Fall back to chemical_validity block if present.
        if "Chemical Validity" not in scores_dict:
            chem = data.get("chemical_validity", {}) or {}
            if isinstance(chem, dict) and "overall_chemical_validity" in chem:
                scores_dict["Chemical Validity"] = safe_float(chem.get("overall_chemical_validity"), 0.0)
            else:
                scores_dict["Chemical Validity"] = 0.0

        results.append({
            "id": idea_id,
            "title": data.get("title") or _infer_title_from_folder(idea_dir),
            "scores_dict": scores_dict,
            "score": data.get("overall_score", data.get("score", 0)),
            "score_file": score_file,
        })

    # Keep natural idea order: idea1, idea2, idea3.
    def _idea_sort_key(item):
        stem = str(item.get("id", ""))
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else 999

    results.sort(key=_idea_sort_key)
    return results


def regenerate_final_radar_from_run_dir(run_dir, output_dir=None):
    run_dir = os.path.normpath(str(run_dir))
    output_dir = os.path.normpath(str(output_dir or run_dir))
    os.makedirs(output_dir, exist_ok=True)

    results = load_final_results_from_run_dir(run_dir)
    if not results:
        raise FileNotFoundError(
            f"No idea*_scores.json files found under: {run_dir}"
        )

    out = plot_final_radar(results, output_dir)
    print(f"Loaded {len(results)} score files.")
    print(f"Saved radar plot: {out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate final_verification_radar.png from an existing BEAVER Design run directory."
    )
    parser.add_argument(
        "run_dir",
        help="Finished Design run directory containing idea*_*/idea*_scores.json files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to run_dir.",
    )
    args = parser.parse_args()

    regenerate_final_radar_from_run_dir(args.run_dir, args.output_dir)
