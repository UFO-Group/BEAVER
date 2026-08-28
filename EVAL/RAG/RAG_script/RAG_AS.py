# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
from datasets import Dataset
from typing import List, Optional, Any, Dict
from tqdm import tqdm

# 🔇 过滤烦人的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 🔥🔥🔥【核心配置】强制使用 HF 国内镜像 🔥🔥🔥
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# ==========================================
# 1. 路径与环境配置
# ==========================================
current_file_path = os.path.abspath(__file__)
eval_dir = os.path.dirname(current_file_path)
agent_dir = os.path.dirname(eval_dir)
project_root = os.path.dirname(os.path.dirname(agent_dir))

if project_root not in sys.path:
    sys.path.append(project_root)

# 📂 输入输出路径
TEST_DATA_PATH = os.path.join(eval_dir, "test_dataset.json")

def choose_retrieval_mode() -> str:
    mapping = {"1": "hybrid", "2": "dense", "3": "bm25"}
    while True:
        print("\n==============================")
        print("🔧 Select Retrieval Mode")
        print("  [1] Hybrid (Dense + BM25 + RRF)")
        print("  [2] Dense-only (Vector)")
        print("  [3] BM25-only (Keyword)")
        print("==============================")
        x = input("Choose (1/2/3): ").strip()
        if x in mapping:
            mode = mapping[x]
            print(f"✅ Retrieval mode = {mode}\n")
            return mode
        print("❌ Invalid input. Please enter 1 / 2 / 3.")
    
# ==========================================
# 2. 导入项目模块
# ==========================================
try:
    from Agent.Agent_Config.agent_config import LLM_MODEL_RAGAS, CORPUS_CONFIG
    from Agent.Agent_Config.deepseek_client import (
        update_client_settings, 
        get_embedding_via_api,
        call_deepseek_llm
    )
    # 导入核心检索与重排序模块
    from Agent.RAG.rag_core import retrieve_evidence
    from Agent.RAG.rag_llm_score import rerank_with_llm_score
    from Agent.RAG.vector_store import load_vector_store
    
    print("✅ 成功导入 Agent 核心组件")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保脚本位于 Agent/EVAL/ 目录下，且项目结构正确。")
    sys.exit(1)

# API 配置
_corpus_levels = {
    str(cfg.get("granularity", "chunk")).strip().lower()
    for cfg in CORPUS_CONFIG
}
if _corpus_levels == {"chunk"}:
    CORPUS_LEVEL = "chunk"
elif _corpus_levels and _corpus_levels <= {"para", "paras", "paragraph", "paragraphs"}:
    CORPUS_LEVEL = "para"
else:
    raise RuntimeError(
        f"Inconsistent corpus granularities in CORPUS_CONFIG: {sorted(_corpus_levels)}"
    )

RAG_ROOT = os.path.dirname(eval_dir)
OUTPUT_ROOT = os.path.join(RAG_ROOT, CORPUS_LEVEL, "RAG_AS")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

USER_API_KEY = os.getenv("RAG_API_KEY")
USER_BASE_URL = os.getenv("RAG_BASE_URL")

if not USER_API_KEY or not USER_BASE_URL:
    raise RuntimeError(
        "Missing RAG API configuration. Set RAG_API_KEY and RAG_BASE_URL "
        "before running this script."
    )

if not update_client_settings(USER_API_KEY, USER_BASE_URL):
    print("❌ 客户端初始化失败，终止程序。")
    sys.exit(1)

# ==========================================
# 3. 第一阶段：数据生成 (修复 Recall=0 的关键)
# ==========================================
def generate_evaluation_data(
    mode: str,
    data_csv_path: str,
    test_data_path: str,
    pool_k: int = 15, # 45
    final_k: int = 2, # 15
):
    """
    重新运行检索和生成，创建新的 CSV 文件（按 mode 保存）。
    """
    print("\n" + "="*60)
    print(f"🚀 [Phase 1] Generating evaluation data | mode={mode} | pool_k={pool_k} | final_k={final_k}")
    print("="*60)

    print("⏳ Loading Vector Store...")
    need_bm25 = mode in ("hybrid", "bm25")
    load_vector_store(build_bm25=need_bm25)

    if not os.path.exists(test_data_path):
        print(f"❌ 找不到测试集: {test_data_path}")
        sys.exit(1)

    with open(test_data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    records = []
    print(f"🧪 开始处理 {len(questions)} 条测试数据...")

    for item in tqdm(questions, desc="Generating RAG Responses"):
        time.sleep(1.5)

        query = item['question']
        ground_truth = item['ground_truth']

        # ✅ 关键：把模式传进 retrieve_evidence（pool 用 45 对齐主流程）
        pool_results = retrieve_evidence(query, top_k=pool_k, retrieval_mode=mode)

        # --- Path A: Raw (Baseline): 直接取前 final_k ---
        raw_docs = pool_results[:final_k]
        ctx_raw_list = [d.get('evidence', '').strip() for d in raw_docs]

        ctx_raw_str = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(ctx_raw_list)])
        prompt_raw = f"You are a scientist. Answer utilizing ONLY the context below:\n{ctx_raw_str}\n\nQuestion: {query}\nAnswer:"
        ans_raw = call_deepseek_llm(prompt_raw)

        # --- Path B: Rerank (Experiment): 从 pool_k 里 rerank 到 final_k ---
        reranked_docs, _ = rerank_with_llm_score(
            query=query,
            results=pool_results,
            top_n=final_k,
            batch_size=5,
            use_batch=True
        )
        ctx_rerank_list = [d.get('evidence', '').strip() for d in reranked_docs]

        ctx_rerank_str = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(ctx_rerank_list)])
        prompt_rerank = f"You are a scientist. Answer utilizing ONLY the context below:\n{ctx_rerank_str}\n\nQuestion: {query}\nAnswer:"
        ans_rerank = call_deepseek_llm(prompt_rerank)

        records.append({
            "question": query,
            "ground_truth": ground_truth,
            "ctx_raw": ctx_raw_list,
            "ans_raw": ans_raw,
            "ctx_rerank": ctx_rerank_list,
            "ans_rerank": ans_rerank
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(data_csv_path), exist_ok=True)
    df.to_csv(data_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Data saved: {data_csv_path}")
    return df

# ==========================================
# 4. LangChain 适配器 (强力修正版)
# ==========================================
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

EMBEDDING_DIM = 1024

class DeepSeekRagasEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            emb = get_embedding_via_api(text)
            if isinstance(emb, np.ndarray):
                embeddings.append(emb.tolist())
            elif isinstance(emb, list):
                embeddings.append(emb)
            else:
                embeddings.append([0.0] * EMBEDDING_DIM)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        emb = get_embedding_via_api(text)
        if isinstance(emb, np.ndarray):
            return emb.tolist()
        elif isinstance(emb, list):
            return emb
        return [0.0] * EMBEDDING_DIM

class DeepSeekRagasLLM(ChatOpenAI):
    def __init__(self, **kwargs):
        kwargs["n"] = 1
        kwargs["temperature"] = kwargs.get("temperature", 0.01)
        super().__init__(**kwargs)

    def _sanitize_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["n"] = 1
        for k in ["best_of", "frequency_penalty", "presence_penalty", "logit_bias"]:
            kwargs.pop(k, None)
        if "model_kwargs" in kwargs:
            kwargs["model_kwargs"]["n"] = 1
            for k in ["best_of", "frequency_penalty", "presence_penalty", "logit_bias"]:
                kwargs["model_kwargs"].pop(k, None)
        return kwargs

    def _clean_output_content(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if code_block:
            text = code_block.group(1)
        text = text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            text = text[start : end+1]
        return text

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs = self._sanitize_params(kwargs)
        res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in res.generations:
            gen.text = self._clean_output_content(gen.text)
        return res

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs = self._sanitize_params(kwargs)
        res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in res.generations:
            gen.text = self._clean_output_content(gen.text)
        return res

# ==========================================
# 5. 绘图与报表
# ==========================================
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness

def plot_distributions_2x5(df_raw, df_rerank, save_path):
    """
    绘制 2x5 分布图 - 科研级高颜值版 (KDE + Rug + Fill)
    Row 1: Raw (Baseline)
    Row 2: Rerank (Experiment)
    Cols: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness
    """
    # ============ 🔥🔥🔥 样式配置 ============
    FONT_SCALE = 0.8        
    TITLE_FONT_SIZE = 20    # 标题大小
    LABEL_FONT_SIZE = 20    # X/Y轴名称大小
    MAIN_TITLE_SIZE = 24    # 大标题大小
    
    TICK_FONT_SIZE  = 18    # 刻度线数字大小

    CURVE_LINE_WIDTH = 3.5  # 曲线粗细
    MEAN_LINE_WIDTH  = 3.0  # 均值线粗细
    
    BOX_LINE_WIDTH = 2.5    # 边框粗细
    BOX_COLOR = 'black'     
    # ========================================

    # 背景设置
    sns.set_theme(style="white", context="talk", font_scale=FONT_SCALE)
    
    # 🔥 修改点 1：画布改为 2 行 5 列，并将宽度从 24 增加到 30，防止图片挤在一起
    fig, axes = plt.subplots(2, 5, figsize=(30, 10)) 
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # 颜色定义 (Raw用绿色，Rerank用蓝色)
    color_raw = (126/255, 195/255, 76/255)      
    color_rerank = (98/255,  126/255, 186/255)  

    def draw_hist(ax, data, color, title, xlabel):
        # 数据清洗转数值
        valid = pd.to_numeric(data, errors='coerce').dropna()
        
        if len(valid) > 1:
            # 动态计算 X 轴范围
            d_min, d_max = valid.min(), valid.max()
            padding = (d_max - d_min) * 0.2 if (d_max - d_min) > 0 else 0.05
            x_low = d_min - padding
            x_high = d_max + padding
            
            # 🔥🔥🔥 核心修改：科研级绘图 🔥🔥🔥
            # 1. KDE 密度图 + 填充
            # bw_adjust=0.6: 让峰值更尖锐，避免过度平滑
            sns.kdeplot(valid, color=color, fill=True, alpha=0.3, 
                        linewidth=CURVE_LINE_WIDTH, bw_adjust=0.6, ax=ax)
            
            # 2. Rug Plot 数据须 (底部小短线，展示真实数据点)
            sns.rugplot(valid, color=color, height=0.05, alpha=0.6, ax=ax)
            
            # 3. 均值线 (高亮红线)
            mean_val = valid.mean()
            ax.axvline(mean_val, color='#E63946', linestyle='--', 
                       linewidth=MEAN_LINE_WIDTH, alpha=0.9)
            
            # 标题 (Mean 值显示)
            ax.set_title(f"{title}\n(Mean: {mean_val:.4f})", 
                         fontsize=TITLE_FONT_SIZE, fontweight='bold')
            
            ax.set_xlim(x_low, x_high)
            
        else:
            ax.set_title(f"{title} (No Data)", fontsize=TITLE_FONT_SIZE)
            ax.set_xlim(0, 1)

        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONT_SIZE)
        
        # 刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

        # 黑色加粗边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BOX_COLOR)
            spine.set_linewidth(BOX_LINE_WIDTH)

    # 🔥 修改点 2：指标映射列表追加了第五个指标 'answer_correctness'
    metrics_config = [
        ('faithfulness', 'Faithfulness'),
        ('answer_relevancy', 'Ans Relevancy'),
        ('context_precision', 'Ctx Precision'),
        ('context_recall', 'Ctx Recall'),
        ('answer_correctness', 'Ans Correctness')
    ]

    # === Row 1: Raw Pipeline ===
    for col_idx, (metric_key, metric_name) in enumerate(metrics_config):
        if metric_key in df_raw.columns:
            draw_hist(axes[0, col_idx], df_raw[metric_key], color_raw, 
                      f"Raw - {metric_name}", metric_name)

    # === Row 2: Rerank Pipeline ===
    for col_idx, (metric_key, metric_name) in enumerate(metrics_config):
        if metric_key in df_rerank.columns:
            draw_hist(axes[1, col_idx], df_rerank[metric_key], color_rerank, 
                      f"Rerank - {metric_name}", metric_name)

    # 全局标题
    fig.suptitle(f"RAGAS A/B Test Distribution (Model: {LLM_MODEL_RAGAS})", 
                 fontweight='bold', fontsize=MAIN_TITLE_SIZE, y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 样式调整版绘图已保存至: {save_path}")
    plt.close()
    
def print_and_save_summary(res_raw_df, res_rerank_df, out_dir: str):
    print("\n📊 正在计算最终指标...")
    def get_mean(df, col_name):
        return df[col_name].mean() if col_name in df.columns else 0.0

    comparison = pd.DataFrame({
        "Metric": [
            "Faithfulness",
            "Answer Relevancy",
            "Context Precision",
            "Context Recall",
            "Answer Correctness",
        ],
        "Raw (Baseline)": [
            get_mean(res_raw_df, 'faithfulness'),
            get_mean(res_raw_df, 'answer_relevancy'),
            get_mean(res_raw_df, 'context_precision'),
            get_mean(res_raw_df, 'context_recall'),
            get_mean(res_raw_df, 'answer_correctness'),
        ],
        "Rerank (Experiment)": [
            get_mean(res_rerank_df, 'faithfulness'),
            get_mean(res_rerank_df, 'answer_relevancy'),
            get_mean(res_rerank_df, 'context_precision'),
            get_mean(res_rerank_df, 'context_recall'),
            get_mean(res_rerank_df, 'answer_correctness'),
        ]
    })
    
    def calc_improvement(row):
        base = row['Raw (Baseline)']
        exp = row['Rerank (Experiment)']
        if base == 0: return "N/A"
        diff = (exp - base) / base * 100
        return f"{diff:+.2f}%"

    comparison["Improvement"] = comparison.apply(calc_improvement, axis=1)
    comparison.to_csv(os.path.join(out_dir, "ragas_summary.csv"), index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print(f"🏆 RAGAS A/B 测试报告 (LLM: {LLM_MODEL_RAGAS})")
    print("="*60)
    print(comparison.to_string(index=False))
    print("="*60)

# ==========================================
# 6. 主流程 (逻辑控制中心)
# ==========================================
def run_full_pipeline():
    mode = choose_retrieval_mode()
    POOL_K = 45
    FINAL_K = 15

    mode_dir = mode.capitalize()  # Hybrid / Dense / Bm25
    out_dir = os.path.join(OUTPUT_ROOT, mode_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ✅ mode 目录下文件
    data_csv_path = os.path.join(out_dir, "ragas_eval_dataset.csv")
    raw_detail_path = os.path.join(out_dir, "ragas_raw_detail.csv")
    rerank_detail_path = os.path.join(out_dir, "ragas_rerank_detail.csv")
    dist_plot_path = os.path.join(out_dir, "ragas_distribution_2x4.png")

    print(f"\n📌 mode={mode} | POOL_K={POOL_K} | FINAL_K={FINAL_K}")
    print(f"📂 out_dir={out_dir}")
    print(f"📂 dataset={data_csv_path}")

    df = None

    # --- 1) 数据阶段 ---
    if os.path.exists(data_csv_path):
        print("   -> 发现已有数据文件。")
        user_choice = input(
            "❓ 是否【重新生成】数据？(输入 'y' 重跑生成，输入 'eval' 仅重跑评分，回车直接看结果): "
        ).lower().strip()

        if user_choice == 'y':
            df = generate_evaluation_data(
                mode=mode,
                data_csv_path=data_csv_path,
                test_data_path=TEST_DATA_PATH,
                pool_k=POOL_K,
                final_k=FINAL_K,
            )
        elif user_choice == 'eval':
            print("⏩ 使用现有数据进行重新评分...")
            df = pd.read_csv(data_csv_path)
        else:
            if os.path.exists(raw_detail_path) and os.path.exists(rerank_detail_path):
                print("⏩ 跳过所有计算，直接生成报表...")
                res_raw_df = pd.read_csv(raw_detail_path)
                res_rerank_df = pd.read_csv(rerank_detail_path)
                print_and_save_summary(res_raw_df, res_rerank_df, out_dir=out_dir)
                plot_distributions_2x5(res_raw_df, res_rerank_df, dist_plot_path)
                return
            else:
                print("⚠️ 未找到评分结果，必须运行评分流程。")
                df = pd.read_csv(data_csv_path)
    else:
        print("⚠️ 未找到数据文件，必须先生成数据。")
        df = generate_evaluation_data(
            mode=mode,
            data_csv_path=data_csv_path,
            test_data_path=TEST_DATA_PATH,
            pool_k=POOL_K,
            final_k=FINAL_K,
        )

    # --- 2) CSV 恢复 list（你原来这一段保留） ---
    print("\n🔄 正在准备数据集...")
    if isinstance(df['ctx_raw'].iloc[0], str):
        df['ctx_raw'] = df['ctx_raw'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df['ctx_rerank'] = df['ctx_rerank'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # --- 3) RAGAS Evaluate（你原来这一段保留） ---
    print("\n⚖️ [Phase 2] 开始 RAGAS 评分...")

    ragas_llm = DeepSeekRagasLLM(model=LLM_MODEL_RAGAS, openai_api_key=USER_API_KEY, openai_api_base=USER_BASE_URL)
    ragas_emb = DeepSeekRagasEmbeddings()
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall(), AnswerCorrectness()]
    config = RunConfig(max_workers=1, timeout=240)

    print("   -> 正在评测 Raw Pipeline...")
    ds_raw = Dataset.from_pandas(df[['question', 'ground_truth', 'ans_raw', 'ctx_raw']]
                                .rename(columns={'ans_raw': 'answer', 'ctx_raw': 'contexts'}))
    res_raw = evaluate(ds_raw, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb, run_config=config, raise_exceptions=False)

    print("   -> 正在评测 Rerank Pipeline...")
    ds_rerank = Dataset.from_pandas(df[['question', 'ground_truth', 'ans_rerank', 'ctx_rerank']]
                                   .rename(columns={'ans_rerank': 'answer', 'ctx_rerank': 'contexts'}))
    res_rerank = evaluate(ds_rerank, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb, run_config=config, raise_exceptions=False)

    # --- 4) 保存（按 mode） ---
    res_raw_df = res_raw.to_pandas()
    res_rerank_df = res_rerank.to_pandas()

    res_raw_df.to_csv(raw_detail_path, index=False, encoding='utf-8-sig')
    res_rerank_df.to_csv(rerank_detail_path, index=False, encoding='utf-8-sig')

    print_and_save_summary(res_raw_df, res_rerank_df, out_dir=out_dir)
    plot_distributions_2x5(res_raw_df, res_rerank_df, dist_plot_path)

    print(f"\n✅ Done. Check folder: {out_dir}")

if __name__ == "__main__":
    run_full_pipeline()
