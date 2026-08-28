# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import time
import traceback
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# =========================================================================
# 0. 环境与绘图配置
# =========================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT_LIB = True
except ImportError:
    HAS_PLOT_LIB = False
    print("⚠️ 缺少 matplotlib/seaborn，将跳过绘图步骤。")

# === 路径配置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# === 导入 Agent 核心组件 ===
try:
    from Agent.Agent_Config import agent_config
    from Agent.Agent_Config import deepseek_client as deepseek_client_module
    from Agent.Agent_Config.deepseek_client import update_client_settings
except ImportError as e:
    print(f"❌ 核心组件导入失败: {e}")
    sys.exit(1)

# === 导入 Agent 类 ===
try:
    from Agent.Planner.planner_agent import ResearchAgent
except ImportError:
    try:
        from Agent.research_agent import ResearchAgent
    except ImportError:
        print("❌ 无法找到 ResearchAgent，请检查路径。")
        sys.exit(1)

# =========================================================================
# 1. 路径与运行配置
# =========================================================================
TEST_DATA_PATH = os.path.join(current_dir, "single_singlewithfacets_multipart_dataset.json")
JUDGE_OUTPUT_DIR = os.path.join(os.path.dirname(current_dir), "Judge_output")
ARCHIVE_DIR = os.path.join(JUDGE_OUTPUT_DIR, "0723-后续返稿用这个")
OUTPUT_DIR = os.path.join(ARCHIVE_DIR, "Forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_ANS_PATH = os.path.join(ARCHIVE_DIR, "1_raw_answers.csv")
GEN_STATUS_PATH = os.path.join(OUTPUT_DIR, "1b_generation_status_forward.csv")
ITEM_SCORE_PATH = os.path.join(OUTPUT_DIR, "2_item_scores_forward.csv")
JUDGE_PARSE_FAIL_PATH = os.path.join(OUTPUT_DIR, "2b_judge_parse_failures_forward.csv")
JUDGE_REPAIR_MAX_ROUNDS = int(os.getenv("JUDGE_REPAIR_MAX_ROUNDS", "2"))
FINAL_SCORE_PATH = os.path.join(OUTPUT_DIR, "3_final_scores_forward.csv")
FINAL_SCORE_BY_TYPE_PATH = os.path.join(OUTPUT_DIR, "3b_final_scores_by_type_forward.csv")
FINAL_OVERALL_BY_TYPE_MATRIX_PATH = os.path.join(OUTPUT_DIR, "3c_overall_by_question_type_forward.csv")
FINAL_OVERALL_BY_STRUCTURE_MATRIX_PATH = os.path.join(OUTPUT_DIR, "3d_overall_by_question_structure_forward.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "4_arena_chart.png")
RUN_LOG_PATH = os.path.join(OUTPUT_DIR, "0_run_log_forward.txt")
BEAVER_RUN_DIR = os.path.join(OUTPUT_DIR, "beaver_runs_forward")
os.makedirs(BEAVER_RUN_DIR, exist_ok=True)

MAX_WORKERS_DIRECT = 2
MAX_WORKERS_JUDGE = 2
MAX_WORKERS_BEAVER = 1   # BEAVER 仍保持串行

SAVE_EVERY_N_DIRECT = 5
SAVE_EVERY_N_BEAVER = 1
SAVE_EVERY_N_JUDGE = 10

QUESTION_STRUCTURE_LABELS = {"single", "single_with_facets", "multi_part"}
QUESTION_TASK_LABELS = {
    "factual_recall", "definition", "mechanism", "comparison",
    "design", "troubleshooting", "synthesis"
}
META_EXCLUDE_COLUMNS = ["question", "ground_truth", "id", "question_type", "question_structure", "topic"]

METRIC_COLUMNS = [
    "Detail_Specificity",
    "Scientific_Grounding",
    "Mechanistic_Explanation",
    "Organization_Format_Quality",
    "Anchor_Alignment",
    "Internal_Consistency",
    "Research_Utility",
]

ITEM_SCORE_COLUMNS = ["row_id", "Model"] + METRIC_COLUMNS + ["JudgeParseOK", "Critique"]


# =========================================================================
# 1.1 BEAVER 评测配置（仅保留 NoQualityLoop）
# =========================================================================
# 当前脚本只保留网页默认态：
# - BEAVER_NoQualityLoop：QL=OFF, STM=OFF, LTM=OFF
BEAVER_VARIANTS = [
    {
        "name": "BEAVER_NoQualityLoop",
        "enable_quality_loop": False,
        "use_short_term": False,
        "use_long_term": False,
        "save_subdir": "no_quality_loop",
        "description": "Web-aligned default BEAVER without quality loop"
    }
]

BEAVER_MODEL_COLUMNS = [cfg["name"] for cfg in BEAVER_VARIANTS]

# 与网页端 chat.py 保持一致的网络重试参数
BEAVER_MAX_RETRIES = 5
BEAVER_RETRY_WAIT_SEC = 5
NETWORK_ERROR_KEYWORDS = [
    "timeout", "connection", "connect", "proxy", "handshake",
    "remote end closed", "empty response", "429", "500", "502", "503", "504"
]

# =========================================================================
# 2. API 配置
# =========================================================================
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "your_api_url")

def env_or_default(key, default=""):
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else val

# 直接对手：闭卷 Direct LLM
OPPONENT_REGISTRY = {
    "Constrained_DeepSeek-V3": {
        "model_name": "DeepSeek-V3.2",
        "api_key": "your_api_key",
        "base_url": "your_api_url",
        "mode": "constrained"
    },
    "Raw_DeepSeek-V3": {
        "model_name": "DeepSeek-V3.2",
        "api_key": "your_api_key",
        "base_url": "your_api_url",
        "mode": "raw"
    },
    "Constrained_DeepSeek-R1": {
        "model_name": "DeepSeek-R1",
        "api_key": "your_api_key",
        "base_url": "your_api_url",
        "mode": "constrained"
    },
    "Raw_DeepSeek-R1": {
        "model_name": "DeepSeek-R1",
        "api_key": "your_api_key",
        "base_url": "your_api_url",
        "mode": "raw"
    },
}

# Judge
JUDGE_CONFIG = {
    "model_name": env_or_default("JUDGE_MODEL", "DeepSeek-R1"),
    "api_key": env_or_default("JUDGE_API_KEY", "your_api_key"),
    "base_url": env_or_default("JUDGE_BASE_URL", DEFAULT_BASE_URL),
}

# =========================================================================
# 3. 工具函数
# =========================================================================
def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")
    
def normalize_question_structure(raw_value, question=""):
    value = "" if raw_value is None else str(raw_value).strip().lower()
    if value in QUESTION_STRUCTURE_LABELS:
        return value
    q = " ".join(str(question).lower().split())
    if not q:
        return "single"
    explicit_multi_patterns = [
        r"\b(first|second|third)\b",
        r"\bpart\s*[12]\b",
        r"\b(then|and then|followed by)\b",
        r"[?].+[?]",
        r"[;；].+[?？]?",
    ]
    if any(re.search(p, q) for p in explicit_multi_patterns):
        return "multi_part"
    facet_markers = [
        "considering", "in terms of", "with respect to", "among", "across",
        "trade-off", "tradeoff", "balance", "simultaneously", "while considering",
        "when considering", "from the perspectives of", "兼顾"
    ]
    comma_count = q.count(",") + q.count("，")
    if any(m in q for m in facet_markers) or comma_count >= 2:
        return "single_with_facets"
    return "single"

def normalize_dataset_question_type(raw_value, question=""):
    value = "" if raw_value is None else str(raw_value).strip().lower()
    alias_map = {
        "fact": "factual_recall",
        "factual": "factual_recall",
        "recall": "factual_recall",
        "definition": "definition",
        "mechanism": "mechanism",
        "comparison": "comparison",
        "design": "design",
        "troubleshooting": "troubleshooting",
        "troubleshoot": "troubleshooting",
        "synthesis": "synthesis",
    }
    if value in QUESTION_TASK_LABELS:
        return value
    if value in alias_map:
        return alias_map[value]
    return infer_question_type(question)


def build_base_results_df(data_items):
    df = pd.DataFrame({
        "id": [d.get("id", "") for d in data_items],
        "question": [d.get("question", "") for d in data_items],
        "ground_truth": [d.get("ground_truth", d.get("answer", "")) for d in data_items],
        "question_structure": [
            normalize_question_structure(d.get("question_structure", d.get("structure", d.get("type", ""))), d.get("question", ""))
            for d in data_items
        ],
        "topic": [d.get("topic", "") for d in data_items],
        "question_type": [
            normalize_dataset_question_type(d.get("question_type", d.get("task_type", "")), d.get("question", ""))
            for d in data_items
        ],
    })
    ensure_result_columns(df)
    return df


def log_msg(msg, also_print=True):
    line = f"[{now_str()}] {msg}"
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(line)

def save_df_atomic(df, path):
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, path)

def preview_text(text, max_len=160):
    s = str(text).replace("\n", " ")
    return s[:max_len] + ("..." if len(s) > max_len else "")

def clean_think_tag(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def normalize_answer(text):
    text = clean_think_tag(text)
    text = text.replace("\u200b", " ").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def is_invalid_answer(text):
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return True

    s_low = s.lower()
    if len(s) < 3:
        return True

    invalid_prefixes = [
        "error:",
        "agent error:",
        "api error:",
        "judge error:",
        "❌ api error:",
        "❌ error:",
        "❌ agent error:",
        "traceback (most recent call last):",
    ]
    if any(s_low.startswith(p) for p in invalid_prefixes):
        return True

    invalid_full_or_near_full_patterns = [
        r"^\s*(request )?timeout\s*$",
        r"^\s*connection refused\s*$",
        r"^\s*connection reset\s*$",
        r"^\s*rate limit exceeded\s*$",
        r"^\s*insufficient quota\s*$",
        r"^\s*please check api quota\s*$",
        r"^\s*重试\s*\d+\s*次后依然失败\s*$",
        r"^\s*http\s*(429|500|502|503|504)\s*$",
    ]
    if any(re.search(p, s_low) for p in invalid_full_or_near_full_patterns):
        return True

    invalid_signal_patterns = [
        r"traceback \(most recent call last\):",
        r"openai\.(?:api|rate|authentication|permission|timeout|connection)?error",
        r"requests\.exceptions\.",
        r"max retries exceeded",
        r"remote end closed connection",
        r"failed to establish a new connection",
        r"temporarily unavailable",
        r"service unavailable",
        r"bad gateway",
        r"gateway timeout",
        r"internal server error",
    ]
    if any(re.search(p, s_low) for p in invalid_signal_patterns):
        return True

    return False

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def needs_generation(val):
    if pd.isna(val):
        return True
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return True
    if is_invalid_answer(s):
        return True
    return False

def ensure_result_columns(df):
    for m in JUDGE_MODEL_ORDER:
        if m not in df.columns:
            df[m] = ""

def load_existing_results(base_df):
    if os.path.exists(RAW_ANS_PATH):
        try:
            old = pd.read_csv(RAW_ANS_PATH)
            for col in old.columns:
                if col in base_df.columns or col in list(OPPONENT_REGISTRY.keys()) + BEAVER_MODEL_COLUMNS:
                    base_df[col] = old[col]
            log_msg(f"✅ 已加载旧 raw answers: {RAW_ANS_PATH}")
        except Exception as e:
            log_msg(f"⚠️ 读取旧 raw answers 失败: {e}")
    ensure_result_columns(base_df)
    return base_df

def build_generation_status(df_results):
    model_cols = [m for m in JUDGE_MODEL_ORDER if m in df_results.columns]
    records = []

    for model in model_cols:
        vals = df_results[model].fillna("").astype(str)
        n_total = len(vals)
        n_error = vals.str.startswith("Error:").sum()
        n_agent_error = vals.str.startswith("Agent Error:").sum()
        n_empty = (vals.str.strip() == "").sum() + (vals.str.strip().str.lower() == "nan").sum()
        n_ok = n_total - n_error - n_agent_error - n_empty
        records.append({
            "Model": model,
            "Total": n_total,
            "OK": int(n_ok),
            "Error": int(n_error),
            "AgentError": int(n_agent_error),
            "Empty": int(n_empty),
            "SuccessRate(%)": round(100.0 * n_ok / max(n_total, 1), 1),
        })

    df_status = pd.DataFrame(records)
    save_df_atomic(df_status, GEN_STATUS_PATH)
    return df_status

def print_generation_status(df_status):
    if df_status.empty:
        return
    log_msg("=== Generation Status ===")
    for _, r in df_status.iterrows():
        log_msg(
            f"{r['Model']}: OK={r['OK']}/{r['Total']}, "
            f"Error={r['Error']}, AgentError={r['AgentError']}, "
            f"Empty={r['Empty']}, SuccessRate={r['SuccessRate(%)']}%"
        )

def load_existing_item_scores():
    if os.path.exists(ITEM_SCORE_PATH):
        try:
            df = pd.read_csv(ITEM_SCORE_PATH)
            missing_metric_cols = [c for c in METRIC_COLUMNS if c not in df.columns]
            if missing_metric_cols:
                log_msg(
                    f"⚠️ 检测到旧版 item scores（缺少新 7 维列: {missing_metric_cols}），将忽略旧评分并要求重新评分。"
                )
                return pd.DataFrame(columns=ITEM_SCORE_COLUMNS)

            for col in ITEM_SCORE_COLUMNS:
                if col not in df.columns:
                    if col in ["row_id", "Model", "Critique"]:
                        df[col] = "" if col == "Critique" else (0 if col == "row_id" else "")
                    else:
                        df[col] = 0
            df = df[ITEM_SCORE_COLUMNS].copy()
            df = purge_removed_model_scores(df)
            log_msg(f"✅ 已加载 item scores: {ITEM_SCORE_PATH}")
            return df
        except Exception as e:
            log_msg(f"⚠️ 读取旧 item scores 失败: {e}")
    return pd.DataFrame(columns=ITEM_SCORE_COLUMNS)


def upsert_item_scores(existing_df, new_records):
    if not new_records:
        return existing_df.copy()

    new_df = pd.DataFrame(new_records)
    if existing_df.empty:
        merged = new_df.copy()
    else:
        merged = existing_df.copy()
        key_pairs = set(zip(new_df["row_id"], new_df["Model"]))
        mask_keep = ~merged.apply(lambda r: (r["row_id"], r["Model"]) in key_pairs, axis=1)
        merged = pd.concat([merged[mask_keep], new_df], ignore_index=True)

    merged = merged.sort_values(by=["Model", "row_id"]).reset_index(drop=True)
    return merged

def drop_item_scores_for_pairs(df_item_scores, model_name, row_ids):
    if df_item_scores.empty or not row_ids:
        return df_item_scores

    mask = ~(
        (df_item_scores["Model"] == model_name) &
        (df_item_scores["row_id"].isin(row_ids))
    )
    return df_item_scores[mask].reset_index(drop=True)
    
def calc_weighted_overall(score_like):
    total = 0.0
    for metric, weight in WEIGHTS.items():
        if isinstance(score_like, dict):
            value = safe_float(score_like.get(metric, 0.0))
        else:
            value = safe_float(score_like[metric])
        total += value * weight
    return total

def build_summary_by_question_type(df_item_scores, df_results):
    if df_item_scores.empty:
        return pd.DataFrame(columns=[
            "QuestionType", "Model", *METRIC_COLUMNS, "Overall", "JudgeParseOK(%)", "ScoredItems", "CoverageWithinType(%)"
        ])

    model_cols = [m for m in JUDGE_MODEL_ORDER if m in df_results.columns]

    meta = df_results.reset_index().rename(columns={"index": "row_id"})[["row_id", "question_type"]]
    merged = df_item_scores.merge(meta, on="row_id", how="left")
    merged["Overall_item"] = merged.apply(calc_weighted_overall, axis=1)

    type_counts = df_results["question_type"].value_counts(dropna=False).to_dict()
    rows = []
    for (qtype, model), sub in merged.groupby(["question_type", "Model"], dropna=False):
        if model not in model_cols or sub.empty:
            continue
        row = {
            "QuestionType": qtype,
            "Model": model,
            "Overall": round(sub["Overall_item"].mean(), 2),
            "JudgeParseOK(%)": round(100.0 * sub["JudgeParseOK"].fillna(0).mean(), 1),
            "ScoredItems": int(len(sub)),
            "CoverageWithinType(%)": round(100.0 * len(sub) / max(type_counts.get(qtype, 1), 1), 1),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = round(sub[metric].mean(), 2)
        rows.append(row)

    df_by_type = pd.DataFrame(rows)
    if not df_by_type.empty:
        df_by_type = df_by_type.sort_values(by=["QuestionType", "Overall"], ascending=[True, False]).reset_index(drop=True)
    return df_by_type

def build_summary_by_question_structure(df_item_scores, df_results):
    if df_item_scores.empty or "question_structure" not in df_results.columns:
        return pd.DataFrame(columns=[
            "QuestionStructure", "Model", *METRIC_COLUMNS, "Overall", "JudgeParseOK(%)", "ScoredItems", "CoverageWithinStructure(%)"
        ])

    model_cols = [m for m in JUDGE_MODEL_ORDER if m in df_results.columns]

    meta = df_results.reset_index().rename(columns={"index": "row_id"})[["row_id", "question_type", "question_structure"]]
    merged = df_item_scores.merge(meta, on="row_id", how="left")
    merged["Overall_item"] = merged.apply(calc_weighted_overall, axis=1)

    structure_counts = df_results["question_structure"].value_counts(dropna=False).to_dict()
    rows = []
    for (qstruct, model), sub in merged.groupby(["question_structure", "Model"], dropna=False):
        total_in_structure = int(structure_counts.get(qstruct, 0))
        scored_items = int(len(sub))
        coverage = round(100.0 * scored_items / max(total_in_structure, 1), 1)
        if model not in model_cols:
            continue
        row = {
            "QuestionStructure": qstruct,
            "Model": model,
            "Overall": round(sub["Overall_item"].mean(), 3),
            "JudgeParseOK(%)": round(100.0 * sub["JudgeParseOK"].fillna(0).mean(), 1),
            "ScoredItems": scored_items,
            "CoverageWithinStructure(%)": coverage,
        }
        for metric in METRIC_COLUMNS:
            row[metric] = round(sub[metric].mean(), 3)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["QuestionStructure", "Overall"], ascending=[True, False]).reset_index(drop=True)

def build_summary_from_item_scores(df_item_scores, df_results):
    if df_item_scores.empty:
        return pd.DataFrame(columns=[
            "Model", *METRIC_COLUMNS, "Overall", "JudgeParseOK(%)",
            "InvalidAnswer(%)", "ScoredItems", "Coverage(%)"
        ])

    model_cols = [m for m in JUDGE_MODEL_ORDER if m in df_results.columns]

    merged = df_item_scores.copy()
    merged["Overall_item"] = merged.apply(calc_weighted_overall, axis=1)

    rows = []
    total_items = len(df_results)

    for model in model_cols:
        sub = merged[merged["Model"] == model]
        if sub.empty:
            continue

        overall_mean = sub["Overall_item"].mean()
        parse_ok_rate = 100.0 * sub["JudgeParseOK"].fillna(0).mean()

        vals = df_results[model].fillna("").astype(str) if model in df_results.columns else []
        invalid_rate = 100.0 * sum(is_invalid_answer(v) for v in vals) / max(len(vals), 1) if len(vals) else 0.0

        row = {
            "Model": model,
            "Overall": round(overall_mean, 2),
            "JudgeParseOK(%)": round(parse_ok_rate, 1),
            "InvalidAnswer(%)": round(invalid_rate, 1),
            "ScoredItems": int(len(sub)),
            "Coverage(%)": round(100.0 * len(sub) / max(total_items, 1), 1),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = round(sub[metric].mean(), 2)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(by="Overall", ascending=False).reset_index(drop=True)
    return df_summary

def save_current_outputs(df_results, df_item_scores):
    try:
        df_raw_to_save = build_raw_answers_with_clean_cols(df_results)
        save_df_atomic(df_raw_to_save, RAW_ANS_PATH)
        log_msg("✅ save_current_outputs: raw answers + cleaned columns saved", also_print=False)
    except Exception as e:
        log_msg(f"❌ save_current_outputs: save raw answers failed: {e}")
        raise

    try:
        df_status = build_generation_status(df_results)
        log_msg("✅ save_current_outputs: generation status built", also_print=False)
    except Exception as e:
        log_msg(f"❌ save_current_outputs: build generation status failed: {e}")
        raise

    try:
        save_df_atomic(df_item_scores, ITEM_SCORE_PATH)
        log_msg("✅ save_current_outputs: item scores saved", also_print=False)
    except Exception as e:
        log_msg(f"❌ save_current_outputs: save item scores failed: {e}")
        raise

    try:
        df_summary = build_summary_from_item_scores(df_item_scores, df_results)
        save_df_atomic(df_summary, FINAL_SCORE_PATH)

        df_summary_by_type = build_summary_by_question_type(df_item_scores, df_results)
        save_df_atomic(df_summary_by_type, FINAL_SCORE_BY_TYPE_PATH)

        if not df_summary_by_type.empty:
            df_matrix = df_summary_by_type.pivot(index="QuestionType", columns="Model", values="Overall").reset_index()
        else:
            df_matrix = pd.DataFrame()
        save_df_atomic(df_matrix, FINAL_OVERALL_BY_TYPE_MATRIX_PATH)

        df_summary_by_structure = build_summary_by_question_structure(df_item_scores, df_results)
        if not df_summary_by_structure.empty:
            df_structure_matrix = df_summary_by_structure.pivot(index="QuestionStructure", columns="Model", values="Overall").reset_index()
        else:
            df_structure_matrix = pd.DataFrame()
        save_df_atomic(df_structure_matrix, FINAL_OVERALL_BY_STRUCTURE_MATRIX_PATH)

        log_msg("✅ save_current_outputs: summary saved", also_print=False)
    except Exception as e:
        log_msg(f"❌ save_current_outputs: build/save summary failed: {e}")
        raise

    try:
        if HAS_PLOT_LIB and not df_summary.empty:
            plot_arena_chart(
                df_summary[["Model", *METRIC_COLUMNS, "Overall"]],
                PLOT_PATH
            )
            log_msg("✅ save_current_outputs: plot saved", also_print=False)
    except Exception as e:
        log_msg(f"❌ save_current_outputs: plot failed: {e}")
        raise

    return df_status, df_summary

def clear_old_outputs():
    for p in [RAW_ANS_PATH, GEN_STATUS_PATH, ITEM_SCORE_PATH, JUDGE_PARSE_FAIL_PATH, FINAL_SCORE_PATH, FINAL_SCORE_BY_TYPE_PATH, FINAL_OVERALL_BY_TYPE_MATRIX_PATH, FINAL_OVERALL_BY_STRUCTURE_MATRIX_PATH, PLOT_PATH, RUN_LOG_PATH]:
        if os.path.exists(p):
            os.remove(p)

    import shutil
    if os.path.exists(BEAVER_RUN_DIR):
        shutil.rmtree(BEAVER_RUN_DIR)
    os.makedirs(BEAVER_RUN_DIR, exist_ok=True)

def inspect_progress(df_results, df_item_scores):
    """
    检查当前运行进度。

    重要：
    - judge_rows_complete 只表示 (row_id, Model) 评分行是否齐全；
    - judge_complete 必须同时满足：
        1) 所有 (row_id, Model) 评分行齐全；
        2) 已有评分里没有 JudgeParseOK == 0。
    """
    # 优先使用显式 Judge 模型顺序，避免把 *_clean 或其它辅助列当成模型列
    try:
        model_cols = [m for m in JUDGE_MODEL_ORDER if m in df_results.columns]
    except NameError:
        exclude_cols = META_EXCLUDE_COLUMNS
        model_cols = [c for c in df_results.columns if c not in exclude_cols and "Unnamed" not in c and not c.endswith("_clean")]

    # raw answers 是否完整
    raw_complete = True
    raw_pending = {}
    for model in model_cols:
        pending_idx = [i for i in range(len(df_results)) if needs_generation(df_results.at[i, model])]
        raw_pending[model] = len(pending_idx)
        if pending_idx:
            raw_complete = False

    # judge 评分行是否齐全：按 (row_id, Model) 唯一对判断
    total_needed_scores = len(df_results) * len(model_cols)
    existing_scores = 0
    judge_parse_zero_pairs = 0
    judge_parse_zero_rows = 0
    judge_parse_zero_row_ids = []

    if df_item_scores is not None and not df_item_scores.empty:
        valid_scores = df_item_scores.copy()

        if {"row_id", "Model"}.issubset(set(valid_scores.columns)):
            valid_scores = valid_scores[valid_scores["Model"].isin(model_cols)].copy()
            valid_scores["row_id_num"] = pd.to_numeric(valid_scores["row_id"], errors="coerce")
            valid_scores = valid_scores.dropna(subset=["row_id_num"])
            valid_scores["row_id_num"] = valid_scores["row_id_num"].astype(int)
            valid_scores = valid_scores[
                (valid_scores["row_id_num"] >= 0) &
                (valid_scores["row_id_num"] < len(df_results))
            ].copy()

            # 若同一个 row_id/model 有重复评分，只看最后一次结果
            valid_scores = valid_scores.drop_duplicates(subset=["row_id_num", "Model"], keep="last")
            unique_pairs = set(zip(valid_scores["row_id_num"], valid_scores["Model"]))
            existing_scores = len(unique_pairs)

            # 关键：这里严格遍历 JudgeParseOK，统计 JudgeParseOK == 0
            if "JudgeParseOK" in valid_scores.columns:
                parse_ok_num = pd.to_numeric(valid_scores["JudgeParseOK"], errors="coerce")
                bad = valid_scores[parse_ok_num == 0].copy()
                judge_parse_zero_pairs = int(len(bad))
                judge_parse_zero_row_ids = sorted(bad["row_id_num"].dropna().astype(int).unique().tolist())
                judge_parse_zero_rows = int(len(judge_parse_zero_row_ids))

    judge_rows_complete = (total_needed_scores > 0 and existing_scores >= total_needed_scores)
    judge_complete = judge_rows_complete and (judge_parse_zero_pairs == 0)

    return {
        "model_cols": model_cols,
        "raw_complete": raw_complete,
        "judge_complete": judge_complete,
        "judge_rows_complete": judge_rows_complete,
        "judge_parse_zero_pairs": judge_parse_zero_pairs,
        "judge_parse_zero_rows": judge_parse_zero_rows,
        "judge_parse_zero_row_ids": judge_parse_zero_row_ids,
        "raw_pending": raw_pending,
        "existing_scores": existing_scores,
        "total_needed_scores": total_needed_scores,
    }


# =========================================================================
# 4. Constrained+Raw LLM
# =========================================================================
Constrained_SYSTEM_PROMPT = """You are a distinguished Professor of Biodegradable Polymer Science.

This is a CLOSED-BOOK evaluation setting.

Your task is to answer scientific questions in a clear, rigorous, mechanism-oriented style that matches an academic query-answering format, but ONLY using your internal knowledge.

STRICT RULES:
1. Answer from internal knowledge only.
2. Do NOT claim to have retrieved, reviewed, searched, or verified any literature.
3. Do NOT fabricate references, DOIs, paper titles, author names, datasets, citation markers, or study-specific details.
4. Do NOT output a References section.
5. Do NOT use fake evidence language such as "the literature shows", "multiple studies confirmed", or "reported values include ..." unless you are speaking in clearly general, non-citation-dependent terms.
6. If a specific number, formulation, processing condition, or mechanism detail is uncertain, state the uncertainty explicitly.
7. When exact details are uncertain, prefer cautious qualitative guidance or clearly labeled approximate ranges rather than invented specificity.
8. Prioritize scientifically grounded mechanism explanation over rhetorical polish.
9. Keep the answer moderately detailed and academically structured, but do NOT write like a full review article.
10. Output plain scientific English.
11. Do NOT use HTML tags such as <sub>, <sup>, <i>, <b>, <br>, or similar markup.
12. You may include an equation ONLY if it is truly necessary and you are confident it is scientifically appropriate.
13. If you include an equation, use ONLY standard Markdown math delimiters:
   - inline math: $...$
   - display math: $$...$$
14. Do NOT use \\[...\\] or \\(...\\).
15. After any equation, define variables in plain English.
16. Do NOT present speculative claims as established facts.
17. Do NOT pad the answer with generic filler. Prefer precision, mechanism, and conditional reasoning.

REQUIRED OUTPUT STRUCTURE:
1. Comprehensive Overview
   - Give a direct, academically phrased answer to the question.
   - Summarize the main conclusion first.

2. In-depth Molecular & Scientific Mechanisms
   - Explain the underlying chemistry, physics, structure-property relationships, degradation behavior, transport behavior, or processing-property logic as appropriate.
   - Focus on why and how.

3. Scientific Synthesis & Practical Interpretation
   - Synthesize the major scientific considerations relevant to the question.
   - Discuss key trade-offs, boundary conditions, and practically important implications.
   - You may mention common field-level trends in a general way, but do NOT imply document-specific evidence.

4. Limitations & Uncertainties
   - State what is well established versus what is uncertain, system-dependent, or likely to vary with composition, morphology, molecular weight, crystallinity, additives, environment, or testing conditions.

STYLE TARGET:
- Sound like a strong scientific query answer.
- Be organized, mechanism-first, and useful.
- Do NOT sound like a retrieval-augmented evidence report.
- Do NOT sound like a polished literature review with hidden citations.
"""

Constrained_USER_TEMPLATE = """Answer the following scientific question in the required format.

Question:
{query}

Additional instructions:
- Stay in a closed-book mode.
- Do not cite or mention references.
- Do not imply that you searched the literature.
- Do not invent specific studies, exact reported datasets, or highly specific numerical values unless you are genuinely confident they are standard knowledge.
- Prefer mechanism-grounded explanation and cautious scientific synthesis.
- When uncertain, explicitly say so.
- Keep the response academically structured and reasonably detailed, but not overly long.

Use exactly these section headings:

1. Comprehensive Overview
2. In-depth Molecular & Scientific Mechanisms
3. Scientific Synthesis & Practical Interpretation
4. Limitations & Uncertainties
"""

RAW_SYSTEM_PROMPT = """You are a expert in biodegradable polymer science.

Answer the user's question directly in scientific English.
"""

RAW_USER_TEMPLATE = """{query}"""




def clean_query_style_formatting(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    text = re.sub(r'<\s*sub\s*>(.*?)<\s*/\s*sub\s*>', r'_\1', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*sup\s*>(.*?)<\s*/\s*sup\s*>', r'^\1', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(i|b|br|em|strong)\s*/?>', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)

    text = text.replace("T_g", "Tg")
    return text.strip()

def call_openai_chat(tag, model_name, api_key, base_url, messages, temperature, timeout_sec, row_id, max_attempts=3):
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        start = time.time()
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_sec,
                max_retries=0
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content if response.choices else ""
            content = normalize_answer(content)
            elapsed = time.time() - start

            log_msg(
                f"✅ API成功 [{tag}] row={row_id} model={model_name} "
                f"time={elapsed:.2f}s chars={len(content)} preview={preview_text(content)}",
                also_print=False
            )
            return {
                "ok": True,
                "content": content,
                "error": "",
                "seconds": elapsed,
            }

        except Exception as e:
            elapsed = time.time() - start
            last_err = str(e)
            log_msg(
                f"❌ API失败 [{tag}] row={row_id} model={model_name} "
                f"attempt={attempt}/{max_attempts} time={elapsed:.2f}s error={last_err}",
                also_print=False
            )
            if attempt < max_attempts:
                sleep_sec = min(15, 2 * attempt)
                time.sleep(sleep_sec)

    return {
        "ok": False,
        "content": "",
        "error": last_err,
        "seconds": 0.0,
    }

def get_Constrained_llm_answer_safe(row_id, query, model_key, conf_val, question_type="factual_recall", question_structure="single"):
    if not conf_val.get("api_key"):
        msg = "Missing API key"
        log_msg(f"❌ Direct/Raw缺少API Key row={row_id} model={model_key}: {msg}", also_print=False)
        return model_key, row_id, f"Error: {msg}", False

    mode = str(conf_val.get("mode", "constrained")).strip().lower()

    if mode == "raw":
        system_prompt = RAW_SYSTEM_PROMPT
        user_prompt = RAW_USER_TEMPLATE.format(query=query)
        tag = f"Raw::{model_key}"
    else:
        system_prompt = Constrained_SYSTEM_PROMPT
        user_prompt = Constrained_USER_TEMPLATE.format(query=query)
        tag = f"Direct::{model_key}"

    result = call_openai_chat(
        tag=tag,
        model_name=conf_val["model_name"],
        api_key=conf_val["api_key"],
        base_url=conf_val["base_url"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.05,
        timeout_sec=120.0,
        row_id=row_id,
        max_attempts=3
    )

    if result["ok"]:
        cleaned = clean_query_style_formatting(result["content"])
        return model_key, row_id, cleaned, True

    err = result.get("error", "Unknown API error")
    return model_key, row_id, f"Error: {err}", False

# =========================================================================
# 5. BEAVER Agent（严格对齐网页后端）
# =========================================================================
def setup_beaver_env():
    """
    对齐网页端 modules/engine.py + modules/chat.py 的配置方式：
    1) 只做 update_client_settings(base_url/api_key)
    2) 默认不强行改写各子模块模型名
    3) 如需强制覆盖模型，可通过环境变量 BEAVER_MODEL_OVERRIDE 显式指定
    """
    sys_api_key = "your_api_key"
    sys_base_url = "your_api_url"
    model_override = env_or_default("BEAVER_MODEL_OVERRIDE", "")

    update_ok = update_client_settings(sys_api_key, sys_base_url)

    if model_override:
        for attr in [
            "LLM_MODEL", "DESIGN_MODEL", "PLANNER_Module_MODEL",
            "INTENT_MODEL", "SCORE_MODEL", "REPORT_MODEL"
        ]:
            if hasattr(deepseek_client_module, attr):
                setattr(deepseek_client_module, attr, model_override)
        if hasattr(agent_config, "LLM_MODEL"):
            agent_config.LLM_MODEL = model_override

    effective_model = model_override or getattr(agent_config, "LLM_MODEL", "(from project config)")
    variant_desc = "; ".join([
        f"{cfg['name']}(QL={cfg['enable_quality_loop']}, STM={cfg['use_short_term']}, LTM={cfg['use_long_term']})"
        for cfg in BEAVER_VARIANTS
    ])
    log_msg(
        f"✅ BEAVER环境已设置(网页对齐): model={effective_model}, "
        f"base_url={sys_base_url}, api_key_present={bool(sys_api_key)}, update_ok={update_ok}, "
        f"variants=[{variant_desc}]"
    )


def build_beaver_agent_web_aligned():
    """
    严格对齐网页端 modules/engine.py 的初始化：
        ResearchAgent(use_memory=True, enable_short_term=True, enable_long_term=True)
    之后每次 run_one_step 再通过 use_short_term/use_long_term 动态控制是否启用。
    """
    return ResearchAgent(
        use_memory=True,
        enable_short_term=True,
        enable_long_term=True
    )

def run_beaver_like_web(agent, query, save_dir, row_id, variant_cfg):
    """
    接收 variant_cfg，动态控制 quality_loop 和 memory 开关
    """
    retry_count = 0
    while True:
        try:
            return agent.run_one_step(
                query,
                enable_quality_loop=variant_cfg.get("enable_quality_loop", False),
                use_short_term=variant_cfg.get("use_short_term", False),
                use_long_term=variant_cfg.get("use_long_term", False),
                save_path=save_dir
            )
        except Exception as e:
            error_str = str(e).lower()
            is_network_error = any(kw in error_str for kw in NETWORK_ERROR_KEYWORDS)
            if is_network_error:
                retry_count += 1
                if retry_count > BEAVER_MAX_RETRIES:
                    raise Exception(
                        f"BEAVER网络/API重试超过上限({BEAVER_MAX_RETRIES})，最后错误: {e}"
                    )
                log_msg(
                    f"⚠️ BEAVER网络错误 row={row_id} retry={retry_count}/{BEAVER_MAX_RETRIES}: {e}. "
                    f"{BEAVER_RETRY_WAIT_SEC}s 后重试。",
                    also_print=False
                )
                time.sleep(BEAVER_RETRY_WAIT_SEC)
                continue
            raise


def get_beaver_answer_task(agent, row_id, query, variant_cfg, question_type="factual_recall", question_structure="single"):
    """
    增加接收 variant_cfg / question_type / question_structure 参数
    """
    query = "" if query is None else str(query)
    safe_query = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", query).strip("_")[:60]
    folder_name = f"q_{row_id:03d}" + (f"_{safe_query}" if safe_query else "")

    # 根据变体的 save_subdir 将不同版本的运行记录分开存放
    subdir = variant_cfg.get("save_subdir", variant_cfg["name"])
    save_dir = os.path.join(BEAVER_RUN_DIR, subdir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        start = time.time()
        # 把 variant_cfg 传给底层执行函数
        ans = run_beaver_like_web(
            agent=agent,
            query=query,
            save_dir=save_dir,
            row_id=row_id,
            variant_cfg=variant_cfg
        )
        ans = normalize_answer(ans)
        elapsed = time.time() - start

        log_msg(
            f"✅ {variant_cfg['name']} 成功 row={row_id} time={elapsed:.2f}s "
            f"chars={len(ans)} preview={preview_text(ans)}",
            also_print=False
        )
        return row_id, ans, True

    except Exception as e:
        elapsed = time.time() - start if "start" in locals() else 0.0
        err = f"{e}"
        log_msg(
            f"❌ {variant_cfg['name']} 失败 row={row_id} time={elapsed:.2f}s error={err}\n{traceback.format_exc()}",
            also_print=False
        )
        return row_id, f"Agent Error: {err}", False

# =========================================================================
# 6. Judge：更公平的评分标准
# =========================================================================
import re
import math

def normalize_for_judge(text: str) -> str:
    """
    仅供 Judge 使用：
    - 保留原始 raw answer 不变
    - 去掉/弱化 markdown 与 LaTeX 源码噪音
    - 尽量把公式转成可读的 plain text，而不是网页渲染格式
    """
    if text is None:
        return ""

    # 处理 pandas 读出来的 nan
    try:
        if isinstance(text, float) and math.isnan(text):
            return ""
    except Exception:
        pass

    s = str(text)

    # --------------------------------------------------
    # 0. 去掉 think 标签 / 代码块围栏
    # --------------------------------------------------
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"```(?:json|markdown|md|text)?", " ", s, flags=re.IGNORECASE)
    s = s.replace("```", " ")

    # --------------------------------------------------
    # 1. 处理常见块公式：$$...$$, \[...\]
    #    这里不直接删除，而是尽量保留文字信息
    # --------------------------------------------------
    s = re.sub(
        r"\$\$(.*?)\$\$",
        lambda m: "\n[Equation] " + m.group(1).strip() + " [/Equation]\n",
        s,
        flags=re.DOTALL
    )
    s = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "\n[Equation] " + m.group(1).strip() + " [/Equation]\n",
        s,
        flags=re.DOTALL
    )

    # --------------------------------------------------
    # 2. 行内公式：$...$
    # --------------------------------------------------
    s = re.sub(r"\$(.*?)\$", lambda m: " " + m.group(1).strip() + " ", s, flags=re.DOTALL)
    s = re.sub(r"\\\((.*?)\\\)", lambda m: " " + m.group(1).strip() + " ", s, flags=re.DOTALL)

    # --------------------------------------------------
    # 3. 常见 LaTeX 命令转普通文本
    # --------------------------------------------------
    latex_map = {
        r"\\alpha": "alpha",
        r"\\beta": "beta",
        r"\\gamma": "gamma",
        r"\\delta": "delta",
        r"\\Delta": "Delta",
        r"\\epsilon": "epsilon",
        r"\\varepsilon": "epsilon",
        r"\\theta": "theta",
        r"\\lambda": "lambda",
        r"\\mu": "mu",
        r"\\nu": "nu",
        r"\\pi": "pi",
        r"\\rho": "rho",
        r"\\sigma": "sigma",
        r"\\tau": "tau",
        r"\\phi": "phi",
        r"\\chi": "chi",
        r"\\omega": "omega",
        r"\\infty": "infinity",
        r"\\approx": "approximately",
        r"\\sim": "approximately",
        r"\\times": "x",
        r"\\cdot": "*",
        r"\\pm": "+/-",
        r"\\leq": "<=",
        r"\\geq": ">=",
        r"\\neq": "!=",
        r"\\to": "->",
        r"\\rightarrow": "->",
        r"\\left": "",
        r"\\right": "",
        r"\\quad": " ",
        r"\\qquad": " ",
        r"\\,": " ",
        r"\\;": " ",
        r"\\:": " ",
        r"\\!": "",
        r"\\%": "%",
    }
    for k, v in latex_map.items():
        s = re.sub(k, v, s)

    # --------------------------------------------------
    # 4. \text{...}, \mathrm{...}, \mathbf{...} 取内容
    # --------------------------------------------------
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathit\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", s)

    # --------------------------------------------------
    # 5. 把 \frac{a}{b} 变成 (a)/(b)
    #    做几轮，处理简单嵌套
    # --------------------------------------------------
    for _ in range(5):
        new_s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s

    # --------------------------------------------------
    # 6. 上下标简单规整
    # --------------------------------------------------
    s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)
    s = re.sub(r"\^\{([^{}]+)\}", r"^\1", s)

    # --------------------------------------------------
    # 7. markdown 清理
    # --------------------------------------------------
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)  # 标题
    s = s.replace("**", "")
    s = s.replace("__", "")
    s = s.replace("* ", "- ")
    s = s.replace("`", "")

    # 列表项统一
    s = re.sub(r"^\s*[-*+]\s*", "- ", s, flags=re.MULTILINE)

    # --------------------------------------------------
    # 8. 去掉残余反斜杠
    # --------------------------------------------------
    s = s.replace("\\(", "(").replace("\\)", ")")
    s = s.replace("\\", " ")

    # --------------------------------------------------
    # 9. 空白折叠
    # --------------------------------------------------
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()

def build_raw_answers_with_clean_cols(df_results):
    """
    仅用于保存 1_raw_answers.csv：
    - 不改动内存中的 df_results
    - 在导出副本末尾追加 *_clean 列
    """
    df_out = df_results.copy()

    # 先加元信息清洗列
    if "question" in df_out.columns:
        df_out["question_clean"] = df_out["question"].apply(normalize_for_judge)
    if "ground_truth" in df_out.columns:
        df_out["ground_truth_clean"] = df_out["ground_truth"].apply(normalize_for_judge)

    # 模型原始列：只认真正的模型列，不把 *_clean 再次当模型
    model_cols = [c for c in list(OPPONENT_REGISTRY.keys()) + BEAVER_MODEL_COLUMNS if c in df_out.columns]

    for model in model_cols:
        df_out[f"{model}_clean"] = df_out[model].apply(normalize_for_judge)

    return df_out

def refresh_clean_answers_csv(df_results):
    """
    只刷新 1_raw_answers.csv 里的 clean 列，不改 raw answer 本体。
    用于“保留 raw answers，重新评分”前，先把导出文件里的 *_clean 列更新。
    """
    df_raw_to_save = build_raw_answers_with_clean_cols(df_results)
    save_df_atomic(df_raw_to_save, RAW_ANS_PATH)
    log_msg("✅ 已重新清洗 raw answers 并覆盖 1_raw_answers.csv")
    
def extract_json_block(text):
    text = clean_think_tag(text)

    block = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1)

    block = re.search(r"JSON\s*:\s*(\{.*?\})", text, flags=re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1)

    block = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if block:
        return block.group(0)

    return None

JUDGE_PROMPT_TEMPLATE = r"""
You are a harsh but calibrated reviewer for materials science QA evaluation.

Your task is to evaluate candidate responses to a materials-science question.
Use the Ground Truth (GT) only as an essential anchor for the core scientific conclusion(s), not as a preferred writing template, preferred length, preferred structure, or upper bound on answer richness.

A strong response may be longer and more detailed than the GT, and may include mechanisms, equations, concrete data, trade-offs, or practical guidance when they genuinely improve the answer.
Do NOT penalize a response merely because it is more detailed or more clearly structured than the GT.
However, literature-like tone, report-like formatting, references, and extended practical discussion should not receive extra credit by themselves unless they materially improve scientific correctness, alignment, or usefulness for the specific question.

Question Type:
{question_type}

Question Structure:
{question_structure}

Question:
{question}

Ground Truth:
{ground_truth}

Candidate Responses:
{candidates_block}

General rules:
1. Evaluate scientific content, scientific usefulness, and research value, not writing flourish alone.
2. Do NOT reward brevity by itself.
3. Do NOT punish longer answers merely for including extra plausible and relevant scientific detail.
4. Treat the GT as an essential anchor for the core conclusion(s). First check whether each response correctly addresses the actual question and captures the GT’s essential scientific point(s).
5. Do NOT require any response to mirror the GT’s wording, structure, level of detail, or length.
6. If a response preserves the GT’s essential conclusion while adding scientifically grounded mechanisms, concrete conditions, useful interpretation, traceable evidence, or research-useful elaboration, reward it.

7. Regarding citations, DOIs, references, evidence IDs, equations, and quantitative data:
   - Do NOT penalize them simply because they are absent from the GT.
   - Real, verifiable references are highly valuable in research and should be rewarded when they genuinely support the claims.
   - Nonstandard but traceable retrieval-style evidence identifiers are acceptable scientific support if they are clearly connected to the claim.
   - Do NOT assume a reference is fabricated merely because it is not formatted as a conventional journal citation.
   - However, references or numbers that do not actually support the stated conclusion should not receive extra credit.

8. Reward precision over vagueness. In materials science, exact parameters, mechanisms, or conditions are better than generic hedging when specificity is possible.
9. Do NOT reward evasive answers that rely heavily on words such as "may", "might", "likely", "possibly", or "depends" when the question calls for a more definite scientific answer.

10. The Response and GT may contain raw Markdown or raw LaTeX-like source text from logs/files rather than rendered formatting.
Treat equations as scientific content, not formatting noise.
Do NOT lower any score merely because an answer contains raw LaTeX or Markdown math syntax such as $, $$, \frac, \text{{}}, \left, \right, \[ \], \( \), subscripts, superscripts, or unrendered symbols.
Judge formulae by their scientific meaning and relevance only.
A formula-heavy answer should NOT be treated as vague, incomplete, or low-quality solely because it is symbolic.
Only penalize when the mathematical content itself is scientifically wrong, irrelevant, contradictory to the GT, or clearly used to evade explanation.

11. Do NOT penalize report-style organization such as section headings, bullet lists, divider lines, numbered sections, bold subsection labels, or structured report formatting.
Markers like ###, -, ---, numbered sections, and bolded subsection labels should be treated as neutral or mildly positive organizational signals unless they genuinely obscure meaning.
Only penalize formatting when it truly harms comprehension, such as unresolved placeholders, broken sentences, malformed equations that prevent interpretation, or clearly incomplete template remnants.

12. For comparison, design, troubleshooting, and synthesis questions:
   - reward actionable reasoning, explicit trade-offs, constraints, recommendation logic, and next-step guidance.
   - a concise but generic answer should score lower on Research Utility than a grounded, more actionable one.

13. For factual_recall and definition questions:
   - prioritize core correctness, calibration, and anchor alignment;
   - do not reward verbosity by itself;
   - but do reward correctly recalled specific scientific details, concrete data, or precise constraints when they are relevant.

14. Use the full 0-10 scale aggressively. A merely plausible answer is NOT an 8+.
15. Be willing to separate close answers by 0.5-1.5 points when one answer is materially more specific, constrained, actionable, or scientifically grounded than another.
16. Avoid position bias, name bias, and length bias. Judge the scientific merit of the content.

You must score each response on the following 7 dimensions from 0.0 to 10.0 with one decimal place:

1) Detail / Specificity
Judge whether the answer is sufficiently detailed, concrete, and informative rather than generic.
- 9-10: Richly detailed, highly specific, and directly informative.
- 7-8.9: Good level of useful specificity.
- 5-6.9: Some detail, but still generic or only moderately informative.
- 0-4.9: Vague, superficial, or largely generic.

2) Scientific Grounding
Judge whether the answer is scientifically justified, well reasoned, and appropriately supported.
Important:
- do not reward references, equations, or numbers by their mere presence;
- reward whether they genuinely strengthen the scientific claim;
- unsupported specificity should reduce the score;
- nonstandard evidence formatting is neutral unless it clearly improves support.
- 9-10: Strongly grounded in credible scientific reasoning or support; evidence/data, if present, clearly strengthens the claims.
- 7-8.9: Mostly well grounded, with meaningful scientific justification.
- 5-6.9: Some scientific basis, but support is thin, weakly connected, or only partly convincing.
- 0-4.9: Little scientific grounding, clearly unsupported claims, or misleading support.

3) Mechanistic Explanation
Judge whether the answer explains why or how, using meaningful materials-science mechanisms rather than only giving conclusions.
- 9-10: Clear, strong, and scientifically meaningful mechanism-level explanation.
- 7-8.9: Good mechanistic explanation, though not fully developed.
- 5-6.9: Some mechanism mentioned, but shallow or incomplete.
- 0-4.9: Little or no meaningful mechanism; mostly assertion.

4) Organization / Format Quality
Judge whether the response is clearly organized, readable, and professionally structured for scientific use.
This is a lower-priority dimension than scientific content.
Do NOT over-reward polish when substance is weak.
Do NOT punish report-like structure or symbolic formatting by itself.
- 9-10: Very clear, well structured, easy to follow, and professionally organized.
- 7-8.9: Generally clear and well organized.
- 5-6.9: Understandable but somewhat loose, cluttered, or unevenly structured.
- 0-4.9: Poorly organized, confusing, or genuinely hard to use.

5) Anchor Alignment
Judge whether the answer actually addresses the user’s question and captures the GT’s essential core conclusion(s).
This does NOT require matching the GT’s wording or brevity.
- 9-10: Directly answers the question and fully captures the GT’s essential scientific conclusion(s).
- 7-8.9: Mostly aligned, with minor omission or slight drift.
- 5-6.9: Partly aligned but misses an important core point or only partially answers the question.
- 0-4.9: Misses the core conclusion, substantially drifts, or does not actually answer the question.

6) Internal Consistency
Judge whether the answer is self-consistent and logically coherent, without internal contradiction, incompatible claims, or mismatch between reasoning and conclusion.
- 9-10: Fully consistent and logically coherent throughout.
- 7-8.9: Mostly consistent, with only minor tension or ambiguity.
- 5-6.9: Noticeable inconsistency, loose logic, or partially mismatched claims.
- 0-4.9: Major internal contradiction or seriously broken reasoning.

7) Research Utility
Judge whether the answer offers genuinely useful scientific or decision-relevant value beyond the core answer.
This is a supplementary dimension, not a primary one.
Do NOT assign a high score merely because the response sounds practical, extensive, or advisory.
- 9-10: Clearly provides strong additional value for research or decision-making, and this added value is well grounded and relevant.
- 7-8.9: Offers meaningful extra practical or research-useful value.
- 5-6.9: Some extra utility, but limited, generic, or only moderately helpful.
- 0-4.9: Little additional utility beyond the core answer.

Task-type calibration:
- If Question Type is factual_recall or definition:
  prioritize Anchor Alignment, Scientific Grounding, and Internal Consistency;
  do not reward verbosity alone.
- If Question Type is mechanism or comparison:
  reward mechanism, explicit trade-offs, and well-supported scientific distinctions.
- If Question Type is design, troubleshooting, or synthesis:
  reward strong mechanism, useful constraints, trade-offs, concrete decision logic, and research utility.
- For design/troubleshooting/synthesis questions:
  if the answer gives no concrete constraints, no useful mechanism, and no actionable guidance,
  Research Utility should usually not exceed 6.5.
- Unsupported exact numeric or procedural claims presented as facts should reduce Scientific Grounding.
- A response that is long but generic should not score highly on Detail / Specificity.
- A polished format cannot compensate for weak science.

Return ONLY valid JSON in this exact schema:
{{
  "Critique": "brief overall comparative critique",
  "Scores": {{
    "MODEL_NAME_1": {{
      "Detail_Specificity": 0.0,
      "Scientific_Grounding": 0.0,
      "Mechanistic_Explanation": 0.0,
      "Organization_Format_Quality": 0.0,
      "Anchor_Alignment": 0.0,
      "Internal_Consistency": 0.0,
      "Research_Utility": 0.0,
      "Critique": "brief critique for this model"
    }},
    "MODEL_NAME_2": {{
      "Detail_Specificity": 0.0,
      "Scientific_Grounding": 0.0,
      "Mechanistic_Explanation": 0.0,
      "Organization_Format_Quality": 0.0,
      "Anchor_Alignment": 0.0,
      "Internal_Consistency": 0.0,
      "Research_Utility": 0.0,
      "Critique": "brief critique for this model"
    }}
  }}
}}
"""

WEIGHTS = {
    "Detail_Specificity": 0.18,
    "Scientific_Grounding": 0.20,
    "Mechanistic_Explanation": 0.16,
    "Organization_Format_Quality": 0.08,
    "Anchor_Alignment": 0.18,
    "Internal_Consistency": 0.08,
    "Research_Utility": 0.12,
}

# 顺序版（如需使用顺序，将下面列表替换到 JUDGE_MODEL_ORDER）
JUDGE_MODEL_ORDER = [
    "BEAVER_NoQualityLoop",
    "Raw_DeepSeek-V3",
    "Raw_DeepSeek-R1",
    "Constrained_DeepSeek-V3",
    "Constrained_DeepSeek-R1",
]

# JUDGE_MODEL_ORDER = [
#     "Raw_DeepSeek-V3",
#     "Raw_DeepSeek-R1",
#     "Constrained_DeepSeek-V3",
#     "Constrained_DeepSeek-R1",
#     "BEAVER_NoQualityLoop",
# ] #倒序

# 当前脚本允许进入生成、评分、汇总的有效模型。
# 作用：读取旧 CSV 时自动过滤已不属于当前实验的旧模型，避免旧结果混入新汇总。
ACTIVE_MODEL_NAMES = set(JUDGE_MODEL_ORDER)
def purge_removed_model_scores(df_item_scores):
    """删除旧评分表中已经不属于当前实验模型集合的旧模型行。"""
    if df_item_scores is None or df_item_scores.empty or "Model" not in df_item_scores.columns:
        return df_item_scores
    before = len(df_item_scores)
    cleaned = df_item_scores[df_item_scores["Model"].isin(ACTIVE_MODEL_NAMES)].copy()
    removed = before - len(cleaned)
    if removed > 0:
        log_msg(f"🧹 已从旧 item scores 中删除已移除模型评分行: {removed} 条")
    return cleaned.reset_index(drop=True)


def infer_question_type(question: str) -> str:
    q = str(question).lower()

    if any(k in q for k in ["what is", "define", "meaning of", "what does", "区别是什么", "定义"]):
        return "definition"

    if any(k in q for k in ["why", "mechanism", "原因", "机理", "how does", "why does"]):
        return "mechanism"

    if any(k in q for k in ["compare", "difference", "versus", "vs", "区别", "比较"]):
        return "comparison"

    if any(k in q for k in ["how to", "design", "optimize", "improve", "制备", "设计", "优化"]):
        return "design"

    if any(k in q for k in ["troubleshoot", "problem", "failed", "not work", "为什么不行", "排查"]):
        return "troubleshooting"

    if any(k in q for k in ["protocol", "procedure", "steps", "synthesis", "路线", "步骤"]):
        return "synthesis"

    return "factual_recall"


JUDGE_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "than", "then", "they", "them",
    "their", "there", "what", "which", "when", "where", "while", "would", "could", "should", "about",
    "because", "through", "between", "under", "over", "after", "before", "across", "toward", "towards",
    "does", "do", "did", "have", "has", "had", "been", "being", "are", "was", "were", "will", "can",
    "may", "might", "also", "more", "most", "less", "much", "many", "some", "such", "both",
    "make", "makes", "made", "main", "question", "answer", "public", "discourse", "scientific", "positive",
    "narrative", "material", "materials", "polymer", "polymers", "science", "sciences", "effect", "effects",
    "system", "systems", "process", "processes", "using", "used", "user", "query", "response", "ground",
    "truth", "important", "major", "minor", "part", "parts", "facet", "facets"
}


def _content_tokens(text: str):
    s = "" if text is None else str(text)
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-_/%.]*", s)
    tokens = []
    for tok in raw_tokens:
        low = tok.lower()
        if len(low) <= 2 and not tok.isupper():
            continue
        if low in JUDGE_STOPWORDS:
            continue
        tokens.append(low)
    return tokens


def _extract_acronym_expansions(text: str):
    s = "" if text is None else str(text)
    exp = {}
    for phrase, acronym in re.findall(r"([A-Za-z][A-Za-z\-\s]{3,80}?)\s*\(([A-Z]{2,}[A-Za-z0-9\-]*)\)", s):
        exp[acronym.upper()] = re.sub(r"\s+", " ", phrase).strip().lower()
    for acronym, phrase in re.findall(r"\b([A-Z]{2,}[A-Za-z0-9\-]*)\s*\(([^)]+)\)", s):
        phrase = re.sub(r"\s+", " ", phrase).strip().lower()
        if len(phrase) >= 4:
            exp[acronym.upper()] = phrase
    return exp


def _build_focus_terms(question: str, ground_truth: str):
    q_tokens = _content_tokens(question)
    gt_tokens = _content_tokens(ground_truth)
    shared = [t for t in gt_tokens if t in set(q_tokens)]
    focus = []
    for tok in shared + gt_tokens[:20] + q_tokens[:12]:
        if tok not in focus:
            focus.append(tok)
    return focus[:18]


def _token_overlap_ratio(prediction: str, focus_terms):
    if not focus_terms:
        return 1.0
    pred_tokens = set(_content_tokens(prediction))
    if not pred_tokens:
        return 0.0
    return len(pred_tokens.intersection(set(focus_terms))) / max(len(set(focus_terms)), 1)


def _leading_focus_ratio(prediction: str, focus_terms, head_words: int = 60):
    if not focus_terms:
        return 1.0
    head = " ".join(str(prediction).split()[:head_words])
    head_tokens = set(_content_tokens(head))
    if not head_tokens:
        return 0.0
    return len(head_tokens.intersection(set(focus_terms))) / max(len(set(focus_terms)), 1)


def _extract_numbers_with_units(text: str):
    text = "" if text is None else str(text)
    pattern = r"\b\d+(?:\.\d+)?(?:\s*(?:°c|c|k|mpa|gpa|pa|wt%|wt\.%|mol%|at%|%|h|min|mins|hr|hrs|hour|hours|day|days|rpm|nm|um|μm|mm|cm|m|g|mg|kg|ml|l|ph))?\b"
    return re.findall(pattern, text.lower())


def _count_condition_keywords(text: str):
    text = "" if text is None else str(text).lower()
    keywords = [
        "temperature", "temp", "time", "wt%", "wt.%", "concentration", "ratio", "ph",
        "pressure", "rpm", "solvent", "anneal", "dry", "drying", "stir", "mix",
        "casting", "extrusion", "melt", "solution", "cooling", "heating", "hold",
        "aging", "curing", "quenching", "loading", "filler", "composition"
    ]
    return sum(1 for k in keywords if k in text)


def _count_overclaim_terms(text: str):
    text = "" if text is None else str(text).lower()
    overclaims = [
        " always ", " never ", " must ", " guarantee", " guarantees", " guaranteed",
        " completely ", " entirely ", " in all cases", " without any", " prove", " proves"
    ]
    padded = f" {text} "
    return sum(padded.count(t) for t in overclaims)


def _has_uncertainty_label(text: str):
    text = "" if text is None else str(text).lower()
    markers = ["approx", "approximately", "typical", "typically", "roughly", "around", "estimated", "estimate"]
    return any(m in text for m in markers)

def _strip_reference_noise_for_scoring(text: str) -> str:
    s = "" if text is None else str(text)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"(?im)^\s*(<<<.*?>>>|END OF INFORMATION|BEGIN OF INFORMATION)\s*$", "", s)
    return s.strip()

def post_adjust_multimodel_scores(payload, prediction, ground_truth, question_type):
    text = " " + ("" if prediction is None else str(prediction)).lower() + " "
    gt = ("" if ground_truth is None else str(ground_truth)).lower()
    qt = str(question_type).lower()

    hedge_hits = sum(text.count(w) for w in [
        " may ", " might ", " likely ", " possibly ", " depends "
    ])

    has_number = bool(re.search(r"\b\d+(\.\d+)?\b", text))
    has_step_words = len(re.findall(r"\b(first|then|finally|step|procedure)\b", text)) >= 2
    has_constraints = any(k in text for k in [
        "temperature", "temp", "time", "wt%", "wt.%", "concentration", "ratio",
        "ph", "pressure", "rpm", "solvent", "anneal", "dry", "drying", "stir",
        "mix", "casting", "extrusion", "melt", "cooling", "heating", "loading"
    ])

    has_formula = bool(re.search(
        r"(\$.*?\$|\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\\frac|\\sum|\\int|\\alpha|\\beta|\\gamma|[A-Za-z0-9\)\]]\s*=\s*[A-Za-z0-9\(\[])",
        "" if prediction is None else str(prediction),
        flags=re.DOTALL
    ))

    has_reference_signal = bool(re.search(
        r"(doi\s*[:：]?\s*\S+|\breferences?\b|\bjournal\b|\bvol\.\b|\bet al\.\b|\b20\d{2}\b)",
        "" if prediction is None else str(prediction),
        flags=re.IGNORECASE
    ))

    generic_answer = (not has_number) and (not has_step_words) and (not has_constraints) and (not has_formula)

    # design / troubleshooting / synthesis：空泛回答封顶
    if qt in ["design", "troubleshooting", "synthesis"] and generic_answer:
        payload["Research_Utility"] = min(payload["Research_Utility"], 6.2)
        payload["Detail_Specificity"] = min(payload["Detail_Specificity"], 6.8)

    # GT明显具体，但回答全是模糊措辞
    if len(gt) > 80 and hedge_hits >= 3:
        payload["Scientific_Grounding"] = min(payload["Scientific_Grounding"], 6.8)
        payload["Anchor_Alignment"] = min(payload["Anchor_Alignment"], 7.0)

    # mechanism题没有机理链条，也别给太高
    if qt == "mechanism" and generic_answer:
        payload["Mechanistic_Explanation"] = min(payload["Mechanistic_Explanation"], 6.0)

    # 有公式/约束/证据痕迹时，不要轻易把组织和grounding打低
    if has_formula or has_constraints or has_reference_signal:
        payload["Organization_Format_Quality"] = max(payload["Organization_Format_Quality"], 6.5)
        payload["Scientific_Grounding"] = max(payload["Scientific_Grounding"], 6.0)

    for k in METRIC_COLUMNS:
        payload[k] = max(0.0, min(10.0, float(payload[k])))

    return payload
    
    return s.strip()
def build_judge_candidates_block(model_to_prediction):
    parts = []
    for model_name, prediction in model_to_prediction.items():
        parts.append(f"### {model_name}\n{prediction}")
    return "\n\n".join(parts).strip()


def evaluate_question_all_answers_safe(row_id, question, ground_truth, model_to_prediction, question_type, question_structure="single"):
    default_score = {metric: 0.0 for metric in METRIC_COLUMNS}
    default_score.update({
        "JudgeParseOK": 0,
        "Critique": "Judge failed or answer invalid.",
    })

    if not JUDGE_CONFIG.get("api_key"):
        log_msg(f"❌ Judge缺少API Key row={row_id}", also_print=False)
        return row_id, {m: default_score.copy() for m in model_to_prediction.keys()}, False

    question_for_judge = normalize_for_judge(question)
    gt_for_judge = _strip_reference_noise_for_scoring(normalize_for_judge(ground_truth))

    model_to_prediction_for_judge = {}
    for model_name, prediction in model_to_prediction.items():
        if is_invalid_answer(prediction):
            model_to_prediction_for_judge[model_name] = "[INVALID ANSWER]"
        else:
            model_to_prediction_for_judge[model_name] = _strip_reference_noise_for_scoring(
                normalize_for_judge(prediction)
            )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question_type=question_type,
        question_structure=question_structure,
        question=question_for_judge,
        ground_truth=gt_for_judge,
        candidates_block=build_judge_candidates_block(model_to_prediction_for_judge)
    )

    result = call_openai_chat(
        tag="Judge::AllModels",
        model_name=JUDGE_CONFIG["model_name"],
        api_key=JUDGE_CONFIG["api_key"],
        base_url=JUDGE_CONFIG["base_url"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        timeout_sec=120.0,
        row_id=row_id,
        max_attempts=3
    )

    if not result["ok"]:
        return row_id, {m: default_score.copy() for m in model_to_prediction.keys()}, False

    text = result["content"]
    json_str = extract_json_block(text)

    if not json_str:
        log_msg(
            f"❌ Judge解析失败 row={row_id} raw={preview_text(text)}",
            also_print=False
        )
        return row_id, {m: default_score.copy() for m in model_to_prediction.keys()}, False

    try:
        parsed = json.loads(json_str)
        score_map = parsed.get("Scores", {}) if isinstance(parsed, dict) else {}
        payload_map = {}

        for model_name, prediction in model_to_prediction.items():
            if is_invalid_answer(prediction):
                payload_map[model_name] = default_score.copy()
                continue

            scores = score_map.get(model_name, {}) if isinstance(score_map, dict) else {}
            payload = {
                metric: round(max(0.0, min(10.0, safe_float(scores.get(metric, 0)))), 1)
                for metric in METRIC_COLUMNS
            }
            payload.update({
                "JudgeParseOK": 1,
                "Critique": str(scores.get("Critique", parsed.get("Critique", ""))).strip(),
            })
            
            payload = post_adjust_multimodel_scores(
                payload,
                prediction,
                ground_truth,
                question_type
            )
            
            payload_map[model_name] = payload

        for model_name in model_to_prediction.keys():
            if model_name not in payload_map:
                payload_map[model_name] = default_score.copy()

        log_msg(
            f"✅ Judge成功 row={row_id} models={len(payload_map)}",
            also_print=False
        )
        return row_id, payload_map, True

    except Exception as e:
        log_msg(
            f"❌ Judge JSON解析异常 row={row_id} error={e} raw={preview_text(text)}",
            also_print=False
        )
        return row_id, {m: default_score.copy() for m in model_to_prediction.keys()}, False



def _safe_row_id_to_int(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return None


def build_judge_parse_failure_report(df_item_scores, df_results, model_cols):
    """
    只扫描 2_item_scores.csv / df_item_scores 里的 JudgeParseOK 列：
    - 仅当已有评分记录的 JudgeParseOK == 0 时，才加入重评列表；
    - 不把缺失评分行当作 JudgeParseOK=0；缺失评分仍由 Round 3 的 pending_judge_idx 处理。

    返回：
    - df_bad_pairs：JudgeParseOK==0 的 pair 报告
    - repair_row_ids：需要重新提交给 Judge 的 row_id 列表
    """
    columns = ["row_id", "Model", "JudgeParseOK", "Reason", "AnswerPreview"]
    records = []
    repair_row_ids = set()

    if not model_cols or df_item_scores is None or df_item_scores.empty:
        return pd.DataFrame(columns=columns), []

    required_cols = {"row_id", "Model", "JudgeParseOK"}
    if not required_cols.issubset(set(df_item_scores.columns)):
        return pd.DataFrame(columns=columns), []

    scores = df_item_scores.copy()
    scores = scores[scores["Model"].isin(model_cols)].copy()
    scores["row_id_int"] = scores["row_id"].apply(_safe_row_id_to_int)
    scores = scores.dropna(subset=["row_id_int"])
    scores["row_id_int"] = scores["row_id_int"].astype(int)

    # 如果同一个 row_id/model 有重复评分，保留最后一次写入的结果。
    scores = scores.drop_duplicates(subset=["row_id_int", "Model"], keep="last")

    for _, r in scores.iterrows():
        row_id = int(r["row_id_int"])
        model = str(r["Model"])
        if row_id < 0 or row_id >= len(df_results):
            continue
        if model not in df_results.columns:
            continue

        raw_parse_ok = r.get("JudgeParseOK", None)
        try:
            parse_ok = float(raw_parse_ok)
        except Exception:
            # 非数值不等同于 JudgeParseOK=0；这里严格只处理 0。
            continue

        if parse_ok == 0.0:
            answer = df_results.at[row_id, model]
            records.append({
                "row_id": row_id,
                "Model": model,
                "JudgeParseOK": raw_parse_ok,
                "Reason": "JudgeParseOK_zero",
                "AnswerPreview": preview_text(answer, 120),
            })
            repair_row_ids.add(row_id)

    df_bad_pairs = pd.DataFrame(records, columns=columns)
    if not df_bad_pairs.empty:
        df_bad_pairs = df_bad_pairs.sort_values(by=["row_id", "Model"]).reset_index(drop=True)
    return df_bad_pairs, sorted(repair_row_ids)


def drop_item_scores_for_rows(df_item_scores, row_ids, model_cols=None):
    """删除指定 row_id 的旧评分。若 model_cols 给定，则只删除这些模型。"""
    if df_item_scores is None or df_item_scores.empty or not row_ids:
        return df_item_scores

    row_id_set = set(int(x) for x in row_ids)
    work = df_item_scores.copy()
    work["row_id_int"] = work["row_id"].apply(_safe_row_id_to_int)

    mask_target = work["row_id_int"].isin(row_id_set)
    if model_cols is not None:
        mask_target = mask_target & work["Model"].isin(model_cols)

    work = work[~mask_target].drop(columns=["row_id_int"], errors="ignore")
    return work.reset_index(drop=True)


def run_judge_scoring_for_rows(df_results, df_item_scores, model_cols, row_ids, desc="Judge::ByQuestion", log_prefix="评分"):
    """
    对给定 row_id 列表进行 Judge 评分。
    注意：当前 Judge prompt 是同一题所有模型一起评分，所以这里按 row_id 整题重评。
    """
    row_ids = sorted(set(int(x) for x in row_ids))
    if not row_ids:
        return df_item_scores

    new_records = []
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_JUDGE) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_question_all_answers_safe,
                idx,
                df_results.at[idx, "question"],
                df_results.at[idx, "ground_truth"],
                {m: df_results.at[idx, m] for m in model_cols},
                df_results.at[idx, "question_type"],
                df_results.at[idx, "question_structure"] if "question_structure" in df_results.columns else "single"
            ): idx
            for idx in row_ids
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=desc
        ):
            idx = future_to_idx[future]
            try:
                row_id, payload_map, ok = future.result()
            except Exception as e:
                row_id = idx
                payload_map = {
                    m: {**{metric: 0.0 for metric in METRIC_COLUMNS}, "Critique": f"Judge execution failed: {e}", "JudgeParseOK": 0}
                    for m in model_cols
                }
                ok = False

            for m_name in model_cols:
                payload = payload_map.get(
                    m_name,
                    {**{metric: 0.0 for metric in METRIC_COLUMNS}, "Critique": "Judge failed or missing model score.", "JudgeParseOK": 0}
                )
                record = {
                    "row_id": row_id,
                    "Model": m_name,
                    "JudgeParseOK": payload.get("JudgeParseOK", 0),
                    "Critique": payload.get("Critique", "")
                }
                for metric in METRIC_COLUMNS:
                    record[metric] = payload.get(metric, 0)
                new_records.append(record)

            done_count += 1
            if done_count % SAVE_EVERY_N_JUDGE == 0 or done_count == len(row_ids):
                df_item_scores = upsert_item_scores(df_item_scores, new_records)
                new_records = []
                df_status, df_summary = save_current_outputs(df_results, df_item_scores)
                log_msg(
                    f"💾 已中间保存{log_prefix}: {done_count}/{len(row_ids)} "
                    f"(latest_ok={ok})"
                )

    if new_records:
        df_item_scores = upsert_item_scores(df_item_scores, new_records)

    return df_item_scores


def maybe_rejudge_failed_parse_rows(df_results, df_item_scores, model_cols):
    """
    评分后补救步骤：严格遍历 JudgeParseOK，若发现 JudgeParseOK == 0，
    则按这些 0 所属的 row_id 整题重新打分。
    """
    df_bad_pairs, repair_row_ids = build_judge_parse_failure_report(df_item_scores, df_results, model_cols)

    if df_bad_pairs.empty:
        log_msg("✅ JudgeParseOK 检查完成：未发现 JudgeParseOK=0。")
        return df_item_scores

    save_df_atomic(df_bad_pairs, JUDGE_PARSE_FAIL_PATH)
    log_msg(
        f"⚠️ JudgeParseOK 检查发现 0：bad_pairs={len(df_bad_pairs)}, "
        f"need_rejudge_rows={len(repair_row_ids)}. 详情已保存: {JUDGE_PARSE_FAIL_PATH}"
    )

    try:
        choice = input(f"是否对 JudgeParseOK=0 的 {len(repair_row_ids)} 个问题重新打分？[Y/n] 默认 Y: ").strip().lower()
    except EOFError:
        choice = ""

    if choice in {"n", "no", "否", "不"}:
        log_msg("⏩ 用户选择跳过 JudgeParseOK=0 自动重评。")
        return df_item_scores

    for repair_round in range(1, max(JUDGE_REPAIR_MAX_ROUNDS, 1) + 1):
        df_bad_pairs, repair_row_ids = build_judge_parse_failure_report(df_item_scores, df_results, model_cols)
        if df_bad_pairs.empty or not repair_row_ids:
            log_msg(f"✅ JudgeParseOK 重评前检查完成：round={repair_round} 已无 JudgeParseOK=0。")
            break

        save_df_atomic(df_bad_pairs, JUDGE_PARSE_FAIL_PATH)
        log_msg(
            f"🔁 JudgeParseOK=0 自动重评 round={repair_round}/{JUDGE_REPAIR_MAX_ROUNDS}: "
            f"rows={len(repair_row_ids)}, bad_pairs={len(df_bad_pairs)}"
        )

        # 因为 Judge 是同一题所有模型一起评分，所以删除这些 row_id 的整题旧评分后重评。
        df_item_scores = drop_item_scores_for_rows(df_item_scores, repair_row_ids, model_cols=model_cols)
        df_item_scores = run_judge_scoring_for_rows(
            df_results=df_results,
            df_item_scores=df_item_scores,
            model_cols=model_cols,
            row_ids=repair_row_ids,
            desc=f"ReJudge::JudgeParseOK0_R{repair_round}",
            log_prefix=f"JudgeParseOK=0重评 round={repair_round}"
        )
        df_status, df_summary = save_current_outputs(df_results, df_item_scores)

    df_bad_pairs, repair_row_ids = build_judge_parse_failure_report(df_item_scores, df_results, model_cols)
    if df_bad_pairs.empty:
        if os.path.exists(JUDGE_PARSE_FAIL_PATH):
            try:
                os.remove(JUDGE_PARSE_FAIL_PATH)
            except Exception:
                pass
        log_msg("✅ JudgeParseOK=0 自动重评完成：当前已无 JudgeParseOK=0。")
    else:
        save_df_atomic(df_bad_pairs, JUDGE_PARSE_FAIL_PATH)
        log_msg(
            f"⚠️ JudgeParseOK=0 自动重评后仍有 bad_pairs={len(df_bad_pairs)}, "
            f"rows={len(repair_row_ids)}. 可能是 Judge 持续解析失败或该题输出导致反复无法解析。"
        )

    return df_item_scores

# =========================================================================
# 7. 绘图
# =========================================================================
def plot_arena_chart(df_scores, save_path):
    if not HAS_PLOT_LIB or df_scores.empty:
        return

    metric_order = METRIC_COLUMNS + ["Overall"]
    df_melt = df_scores.melt(
        id_vars=["Model"],
        value_vars=metric_order,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=df_melt,
        x="Metric",
        y="Score",
        hue="Model",
        palette="rocket",
        edgecolor="black",
        err_kws={'linewidth': 0},
        order=metric_order
    )

    plt.title("BEAVER_NoQualityLoop vs DeepSeek Direct/Raw LLMs (7-Dimension Fair Closed-Book Evaluation)", fontweight="bold", pad=16)
    plt.ylabel("Score (0-10)", fontweight="bold")
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 11)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log_msg(f"📊 图表已保存: {save_path}", also_print=False)

# =========================================================================
# 8. 主流程
# =========================================================================
def main():
    log_msg("=" * 80)
    log_msg("开始运行 BEAVER_VS_LLMs.py")

    if not os.path.exists(TEST_DATA_PATH):
        log_msg(f"❌ 找不到测试数据: {TEST_DATA_PATH}")
        return

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # 兼容你现在这个 dataset 结构
    if isinstance(raw, dict) and "items" in raw:
        data = raw["items"]
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError(f"Unsupported dataset structure: {type(raw)}")
    
    if not data:
        raise ValueError("Dataset is empty.")
    
    df_results = build_base_results_df(data)
    df_results = load_existing_results(df_results)
    df_item_scores = load_existing_item_scores()

    progress = inspect_progress(df_results, df_item_scores)

    print(f"\n📂 输出目录: {OUTPUT_DIR}")
    print(f"🧾 Raw answers complete: {progress['raw_complete']}")
    print(f"⚖️ Judge rows complete: {progress.get('judge_rows_complete', progress['judge_complete'])}")
    print(f"⚖️ Judge complete(rows + JudgeParseOK): {progress['judge_complete']}")
    print(f"📊 Existing judge rows: {progress['existing_scores']}/{progress['total_needed_scores']}")
    print(f"⚠️ JudgeParseOK=0: {progress.get('judge_parse_zero_pairs', 0)} pairs / {progress.get('judge_parse_zero_rows', 0)} rows")
    if progress.get("judge_parse_zero_row_ids"):
        preview_bad_rows = progress["judge_parse_zero_row_ids"][:20]
        suffix = " ..." if len(progress["judge_parse_zero_row_ids"]) > 20 else ""
        print(f"   Bad row_id: {preview_bad_rows}{suffix}")

    # ========== 智能交互 ==========
    run_generation = True
    run_judging = True
    run_rejudge_parse_zero = True

    if (
        progress["raw_complete"]
        and progress.get("judge_rows_complete", False)
        and progress.get("judge_parse_zero_pairs", 0) > 0
    ):
        print("\n⚠️ 检测到：评分行数量齐全，但存在 JudgeParseOK=0。")
        print("1 = 只对 JudgeParseOK=0 对应的 row_id 重新打分（推荐）")
        print("2 = 保留 raw answers，清空全部评分后重新评分")
        print("3 = 仅重算 summary / 重新画图，不修复 JudgeParseOK=0")
        print("4 = 清空全部，重新生成 + 重新评分")
        choice = input("请选择 [1/2/3/4]，默认 1: ").strip()

        if choice == "2":
            refresh_clean_answers_csv(df_results)
            if os.path.exists(ITEM_SCORE_PATH):
                os.remove(ITEM_SCORE_PATH)
            df_item_scores = pd.DataFrame(columns=ITEM_SCORE_COLUMNS)
            run_generation = False
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("▶️ 用户选择：保留 raw answers，清空全部评分后重新评分。")
        elif choice == "3":
            run_generation = False
            run_judging = False
            run_rejudge_parse_zero = False
            log_msg("▶️ 用户选择：仅重算汇总与画图，不修复 JudgeParseOK=0。")
        elif choice == "4":
            clear_old_outputs()
            df_results = build_base_results_df(data)
            df_item_scores = pd.DataFrame(columns=ITEM_SCORE_COLUMNS)
            run_generation = True
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("🔄 用户选择：清空全部，重新生成 + 重新评分。")
        else:
            run_generation = False
            run_judging = False
            run_rejudge_parse_zero = True
            log_msg("▶️ 用户选择：只对 JudgeParseOK=0 对应的 row_id 重新打分。")

    elif progress["raw_complete"] and progress["judge_complete"]:
        print("\n✅ 检测到：原始答案和评分都已经完整，且没有 JudgeParseOK=0。")
        print("1 = 仅重算 summary / 重新画图")
        print("2 = 保留 raw answers，重新评分")
        print("3 = 清空全部，重新生成 + 重新评分")
        choice = input("请选择 [1/2/3]，默认 1: ").strip()

        if choice == "2":
            refresh_clean_answers_csv(df_results)
            if os.path.exists(ITEM_SCORE_PATH):
                os.remove(ITEM_SCORE_PATH)
            df_item_scores = pd.DataFrame(columns=ITEM_SCORE_COLUMNS)
            run_generation = False
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("▶️ 用户选择：保留 raw answers，重新评分。")
        elif choice == "3":
            clear_old_outputs()
            df_results = build_base_results_df(data)
            df_item_scores = pd.DataFrame(columns=ITEM_SCORE_COLUMNS)
            run_generation = True
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("🔄 用户选择：清空全部，重新生成 + 重新评分。")
        else:
            run_generation = False
            run_judging = False
            run_rejudge_parse_zero = False
            log_msg("▶️ 用户选择：仅重算汇总与画图。")

    elif progress["raw_complete"] and not progress["judge_complete"]:
        print("\n✅ 检测到：raw answers 已完整，但评分不完整或存在 JudgeParseOK=0。")
        print("1 = 跳过生成，直接补齐/重评评分")
        print("2 = 清空评分，跳过生成，先重新清洗 clean 列，再重新评分")
        print("3 = 清空全部，重新生成 + 重新评分")
        choice = input("请选择 [1/2/3]，默认 1: ").strip()

        if choice == "2":
            refresh_clean_answers_csv(df_results)
            if os.path.exists(ITEM_SCORE_PATH):
                os.remove(ITEM_SCORE_PATH)
            df_item_scores = pd.DataFrame(columns=ITEM_SCORE_COLUMNS)
            run_generation = False
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("▶️ 用户选择：跳过生成，先重新清洗 clean 列，再重新评分。")
        elif choice == "3":
            clear_old_outputs()
            df_results = build_base_results_df(data)
            df_item_scores = pd.DataFrame(columns=ITEM_SCORE_COLUMNS)
            run_generation = True
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("🔄 用户选择：清空全部，重新生成 + 重新评分。")
        else:
            run_generation = False
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("▶️ 用户选择：直接从评分开始。")

    elif not progress["raw_complete"]:
        print("\n⚠️ 检测到：raw answers 尚未全部完成。")
        print("1 = 继续断点生成，并在完成后评分")
        print("2 = 清空全部，重新生成 + 重新评分")
        choice = input("请选择 [1/2]，默认 1: ").strip()

        if choice == "2":
            clear_old_outputs()
            df_results = build_base_results_df(data)
            df_item_scores = pd.DataFrame(columns=ITEM_SCORE_COLUMNS)
            run_generation = True
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("🔄 用户选择：清空全部，重新生成 + 重新评分。")
        else:
            run_generation = True
            run_judging = True
            run_rejudge_parse_zero = True
            log_msg("▶️ 用户选择：继续断点生成，并后续评分。")

    if run_generation:
        # ---------------------------------------------------------------------
        # Round 1: Direct LLM generation（按模型顺序跑，跑完一个就保存）
        # ---------------------------------------------------------------------
        log_msg(f"🥊 Round 1: Raw + Constrained Direct LLMs (per-model, workers={MAX_WORKERS_DIRECT})")
    
        for model_name, conf in OPPONENT_REGISTRY.items():
            pending_idx = [i for i in range(len(df_results)) if needs_generation(df_results.at[i, model_name])]
            if not pending_idx:
                log_msg(f"⏩ 跳过 {model_name}: 已全部生成完成")
                continue
                
            df_item_scores = drop_item_scores_for_pairs(df_item_scores, model_name, pending_idx)
            log_msg(f"🚀 开始生成 {model_name}: pending={len(pending_idx)}")
    
            done_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_DIRECT) as executor:
                future_to_idx = {
                    executor.submit(
                    get_Constrained_llm_answer_safe,
                    idx,
                    df_results.at[idx, "question"],
                    model_name,
                    conf,
                    df_results.at[idx, "question_type"],
                    df_results.at[idx, "question_structure"] if "question_structure" in df_results.columns else "single"
                ): idx
                    for idx in pending_idx
                }
    
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc=f"{model_name}"
                ):
                    idx = future_to_idx[future]
                    try:
                        _, row_id, ans, ok = future.result()
                        df_results.at[row_id, model_name] = ans
                    except Exception as e:
                        df_results.at[idx, model_name] = f"Error: {e}"
                        ok = False
    
                    done_count += 1
                    if done_count % SAVE_EVERY_N_DIRECT == 0 or done_count == len(pending_idx):
                        df_status, df_summary = save_current_outputs(df_results, df_item_scores)
                        log_msg(
                            f"💾 已中间保存 {model_name}: {done_count}/{len(pending_idx)} "
                            f"(latest_ok={ok})"
                        )
    
            df_status, df_summary = save_current_outputs(df_results, df_item_scores)
            model_vals = df_results[model_name].fillna("").astype(str)
            n_ok = sum(not needs_generation(v) for v in model_vals)
            n_err = sum(v.startswith("Error:") for v in model_vals)
            log_msg(f"✅ {model_name} 完成: ok={n_ok}, error={n_err}")
            print_generation_status(df_status)
    
        # ---------------------------------------------------------------------
        # Round 2: BEAVER variants generation（每题保存，可续跑）
        # ---------------------------------------------------------------------
        log_msg(f"🛡️ Round 2: BEAVER Variants (workers={MAX_WORKERS_BEAVER})")
        setup_beaver_env()
    
        for variant_cfg in BEAVER_VARIANTS:
            beaver_name = variant_cfg["name"]
            beaver_agent = build_beaver_agent_web_aligned()
    
            pending_idx = [i for i in range(len(df_results)) if needs_generation(df_results.at[i, beaver_name])]
            if not pending_idx:
                log_msg(f"⏩ 跳过 {beaver_name}: 已全部生成完成")
                continue
            
            df_item_scores = drop_item_scores_for_pairs(df_item_scores, beaver_name, pending_idx)
            log_msg(f"🧹 已删除 {beaver_name} 待重生成样本的旧评分: {len(pending_idx)} 条")
    
            log_msg(
                f"🚀 开始生成 {beaver_name}: pending={len(pending_idx)}, "
                f"QL={variant_cfg['enable_quality_loop']}, "
                f"STM={variant_cfg['use_short_term']}, LTM={variant_cfg['use_long_term']}"
            )
            count = 0
            for idx in tqdm(pending_idx, desc=beaver_name):
                row_id, ans, ok = get_beaver_answer_task(
                    beaver_agent,
                    idx,
                    df_results.at[idx, "question"],
                    variant_cfg,
                    df_results.at[idx, "question_type"],
                    df_results.at[idx, "question_structure"] if "question_structure" in df_results.columns else "single"
                )
                df_results.at[row_id, beaver_name] = ans
    
                count += 1
                if count % SAVE_EVERY_N_BEAVER == 0 or count == len(pending_idx):
                    df_status, df_summary = save_current_outputs(df_results, df_item_scores)
                    log_msg(f"💾 已中间保存 {beaver_name}: {count}/{len(pending_idx)} (latest_ok={ok})")
    
            df_status, df_summary = save_current_outputs(df_results, df_item_scores)
            model_vals = df_results[beaver_name].fillna("").astype(str)
            n_ok = sum(not needs_generation(v) for v in model_vals)
            n_err = sum(v.startswith("Agent Error:") for v in model_vals)
            log_msg(f"✅ {beaver_name} 完成: ok={n_ok}, agent_error={n_err}")
            print_generation_status(df_status)
    else:
        log_msg("⏩ 已按用户选择跳过生成阶段。")

    if run_judging:
        # ---------------------------------------------------------------------
        # Round 3: Judging（同一题的所有结果同时评分）
        # ---------------------------------------------------------------------
        log_msg(f"⚖️ Round 3: AI Judge scoring by question (workers={MAX_WORKERS_JUDGE})")

        model_cols = [m for m in JUDGE_MODEL_ORDER if m in df_results.columns]

        scored_row_ids = set()
        if not df_item_scores.empty:
            valid_scores = df_item_scores[df_item_scores["Model"].isin(model_cols)].copy()
            valid_scores["row_id_int"] = valid_scores["row_id"].apply(_safe_row_id_to_int)
            valid_scores = valid_scores.dropna(subset=["row_id_int"])
            score_counts = valid_scores.groupby("row_id_int")["Model"].nunique().to_dict()
            scored_row_ids = {
                int(row_id) for row_id, cnt in score_counts.items()
                if cnt >= len(model_cols)
            }

        pending_judge_idx = [i for i in range(len(df_results)) if i not in scored_row_ids]

        if not pending_judge_idx:
            log_msg("⏩ 跳过评分: 已全部完成")
        else:
            log_msg(f"🚀 开始按问题同时评分: pending={len(pending_judge_idx)}")
            df_item_scores = run_judge_scoring_for_rows(
                df_results=df_results,
                df_item_scores=df_item_scores,
                model_cols=model_cols,
                row_ids=pending_judge_idx,
                desc="Judge::ByQuestion",
                log_prefix="评分"
            )

            df_status, df_summary = save_current_outputs(df_results, df_item_scores)
            log_msg("✅ 按问题同时评分完成")
            if not df_summary.empty:
                log_msg("=== 当前排行榜 ===")
                log_msg("\n" + df_summary.to_string(index=False), also_print=True)

    else:
        log_msg("⏩ 已按用户选择跳过评分阶段。")

    # ---------------------------------------------------------------------
    # Round 3.5: 遍历 2_item_scores.csv 中的 JudgeParseOK，若有 0 则按 row_id 重评
    # ---------------------------------------------------------------------
    model_cols = [m for m in JUDGE_MODEL_ORDER if m in df_results.columns]
    if run_rejudge_parse_zero and model_cols and df_item_scores is not None and not df_item_scores.empty:
        df_item_scores = maybe_rejudge_failed_parse_rows(
            df_results=df_results,
            df_item_scores=df_item_scores,
            model_cols=model_cols
        )
    elif not run_rejudge_parse_zero:
        log_msg("⏩ JudgeParseOK=0 检查/重评已按用户选择跳过。")
    else:
        log_msg("⏩ JudgeParseOK 检查跳过：暂无 item scores 或 model columns。")
        
    # ---------------------------------------------------------------------
    # Round 4: Final Summary
    # ---------------------------------------------------------------------
    df_status, df_summary = save_current_outputs(df_results, df_item_scores)
    df_summary_by_type = build_summary_by_question_type(df_item_scores, df_results)
    df_summary_by_structure = build_summary_by_question_structure(df_item_scores, df_results)

    print("\n" + "=" * 80)
    print("🏆 BEAVER_NoQualityLoop vs DeepSeek Direct/Raw LLMs: Fair Closed-Book Leaderboard")
    print("=" * 80)
    if df_summary.empty:
        print("暂无可用汇总结果。")
    else:
        print(df_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("📚 By Question Type")
    print("=" * 80)
    if df_summary_by_type.empty:
        print("暂无按题型汇总结果。")
    else:
        for qtype in df_summary_by_type["QuestionType"].dropna().unique().tolist():
            print(f"\n--- {qtype} ---")
            print(df_summary_by_type[df_summary_by_type["QuestionType"] == qtype].to_string(index=False))

    print("\n" + "=" * 80)
    print("🧩 By Question Structure")
    print("=" * 80)
    if df_summary_by_structure.empty:
        print("暂无按问题结构汇总结果。")
    else:
        for qstruct in df_summary_by_structure["QuestionStructure"].dropna().unique().tolist():
            print(f"\n--- {qstruct} ---")
            print(df_summary_by_structure[df_summary_by_structure["QuestionStructure"] == qstruct].to_string(index=False))

    log_msg(f"✅ raw answers 已保存: {RAW_ANS_PATH}")
    log_msg(f"✅ generation status 已保存: {GEN_STATUS_PATH}")
    log_msg(f"✅ item scores 已保存: {ITEM_SCORE_PATH}")
    log_msg(f"✅ final scores 已保存: {FINAL_SCORE_PATH}")
    log_msg(f"✅ by-type final scores 已保存: {FINAL_SCORE_BY_TYPE_PATH}")
    log_msg(f"✅ by-type overall matrix 已保存: {FINAL_OVERALL_BY_TYPE_MATRIX_PATH}")
    log_msg(f"✅ by-structure overall matrix 已保存: {FINAL_OVERALL_BY_STRUCTURE_MATRIX_PATH}")
    if HAS_PLOT_LIB:
        log_msg(f"✅ arena chart 已保存: {PLOT_PATH}")

    log_msg("运行结束")
    log_msg("=" * 80)

if __name__ == "__main__":
    main()
