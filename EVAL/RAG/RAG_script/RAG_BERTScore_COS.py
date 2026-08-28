# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# 🔥🔥🔥【核心配置】强制使用 HF 国内镜像 🔥🔥🔥
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_func

# ==========================================
# 1. 路径与环境配置
# ==========================================
current_file_path = os.path.abspath(__file__)
eval_dir = os.path.dirname(current_file_path)
agent_dir = os.path.dirname(eval_dir)
project_root = os.path.dirname(os.path.dirname(agent_dir))

# 📂 定义输出目录 (AB测试版)
# 添加项目根目录到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

# 📦 导入 RAG 核心组件
try:
    from Agent.RAG.rag_core import retrieve_evidence
    from Agent.RAG.rag_llm_score import rerank_with_llm_score
    from Agent.RAG.vector_store import load_vector_store
    from Agent.Agent_Config.agent_config import CORPUS_CONFIG
    from Agent.Agent_Config.deepseek_client import update_client_settings, call_deepseek_llm
    print(f"✅ 成功导入 RAG 核心组件")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
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
OUTPUT_DIR = os.path.join(RAG_ROOT, CORPUS_LEVEL, "BERT_COS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_API_KEY = os.getenv("RAG_API_KEY")
USER_BASE_URL = os.getenv("RAG_BASE_URL")

if not USER_API_KEY or not USER_BASE_URL:
    raise RuntimeError(
        "Missing RAG API configuration. Set RAG_API_KEY and RAG_BASE_URL "
        "before running this script."
    )

# 文件路径
TEST_DATA_FILE_NAME = "test_dataset.json"
TEST_DATA_PATH = os.path.join(eval_dir, TEST_DATA_FILE_NAME)
def choose_retrieval_mode() -> str:
    """
    启动时交互式选择检索模式：
      1 = hybrid
      2 = dense
      3 = bm25
    """
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
# 2. 核心逻辑：A/B 双路 RAG 流水线
# ==========================================
def clean_rag_output(text: str) -> str:
    """清洗 AI 输出，去除引用标记"""
    if not isinstance(text, str): return ""
    if "### References" in text:
        text = text.split("### References")[0]
    return text.strip()

def generate_with_llm(query, context_str):
    """通用生成函数"""
    prompt = f"""You are an expert polymer scientist.
Below are extracted text segments from research papers.

Context Evidence:
{context_str}

Question: {query}

### INSTRUCTIONS:
1. Answer the question using **ONLY** the provided Context Evidence.
2. **CITATION RULE**: Cite the evidence index like [1], [2] at the end of sentences.
3. **STRICTLY FORBIDDEN**: DO NOT generate a "References" list at the end. The system will add the file paths automatically.

Please provide a clear, concise scientific answer in English:
"""

    return call_deepseek_llm(prompt)

def manual_rag_pipeline_ab_test(
    query: str,
    retrieval_mode: str = "hybrid",
    pool_k: int = 15,     # 对齐 rag_answer.retrieve_top_k 默认值
    final_k: int = 2     # 对齐 rag_answer.rerank_top_n / top_k 默认值
):
    """
    执行两条路径的 RAG：
    A. Raw Top-15 (不重排序，直接取检索结果前15)
    B. Rerank Top-15 (重排序后取前15)
    """
    results_pool = retrieve_evidence(query, top_k=pool_k, retrieval_mode=retrieval_mode)
    # --- 1. 基础检索 (Retrieval Pool) ---
    # === Path A: Raw Top-15 (Baseline) ===
    # 直接截取原始检索的前 5 个
    raw_docs = results_pool[:final_k]
    
    pieces_raw = []
    full_ctx_raw_text = ""
    for i, r in enumerate(raw_docs, start=1):
        content = r.get('evidence', '').strip()
        pieces_raw.append(f"Evidence [{i}]:\n{content}")
        full_ctx_raw_text += content + "\n"
    
    ctx_str_raw = "\n\n".join(pieces_raw)
    
    # 生成 A
    ans_raw = generate_with_llm(query, ctx_str_raw)

    # === Path B: Rerank Top-15 (Experiment) ===
    reranked_docs, _ = rerank_with_llm_score(
        query=query,
        results=results_pool,
        top_n=final_k,
        use_batch=True
    )
    
    pieces_rerank = []
    full_ctx_rerank_text = ""
    for i, r in enumerate(reranked_docs, start=1):
        content = r.get('evidence', '').strip()
        pieces_rerank.append(f"Evidence [{i}]:\n{content}")
        full_ctx_rerank_text += content + "\n"
        
    ctx_str_rerank = "\n\n".join(pieces_rerank)

    # 生成 B
    ans_rerank = generate_with_llm(query, ctx_str_rerank)
    
    return {
        "ctx_raw": full_ctx_raw_text,
        "ans_raw": ans_raw,
        "ctx_rerank": full_ctx_rerank_text,
        "ans_rerank": ans_rerank
    }

# ==========================================
# 3. 评估指标计算
# ==========================================
def calculate_metrics(candidates_list, references_list, metric_name="Metric"):
    print(f"\n[{metric_name}] 加载 Embedding 模型...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"[{metric_name}] 计算 Cosine Similarity...")
    cand_embeddings = embedder.encode(candidates_list, convert_to_tensor=True)
    ref_embeddings = embedder.encode(references_list, convert_to_tensor=True)
    
    cosine_vals = []
    for c_emb, r_emb in zip(cand_embeddings, ref_embeddings):
        score = util.cos_sim(c_emb, r_emb).item()
        cosine_vals.append(score)

    print(f"[{metric_name}] 计算 BERTScore (F1)...")
    try:
        # 显存优化：batch_size 设小一点
        P, R, F1 = bert_score_func(
            candidates_list, 
            references_list, 
            model_type="roberta-base", 
            lang="en", 
            verbose=True,
            batch_size=4
        )
        bert_vals = F1.numpy().tolist()
    except Exception as e:
        print(f"❌ BERTScore 计算失败: {e}")
        bert_vals = [0.0] * len(candidates_list)

    return cosine_vals, bert_vals

# ==========================================
# 4. 2x4 绘图函数 (刻度字体可调版)
# ==========================================
def plot_distributions_2x4(df, save_path):
    """
    绘制 2x4 分布图 - 标题单行显示 + 无网格 + 黑边框 + 刻度大小可调
    """
    # ============ 🔥🔥🔥 样式配置 ============
    FONT_SCALE = 0.8        
    TITLE_FONT_SIZE = 20    # 标题大小
    LABEL_FONT_SIZE = 20    # X/Y轴名称大小 ("Cosine", "Density")
    MAIN_TITLE_SIZE = 24    # 大标题大小
    
    TICK_FONT_SIZE  = 18    # 🔥🔥🔥 [新增] 刻度线数字大小 (0.2, 0.4...) 🔥🔥🔥

    CURVE_LINE_WIDTH = 3.5  # 曲线粗细
    MEAN_LINE_WIDTH  = 3.0  # 均值线粗细
    
    BOX_LINE_WIDTH = 2.5    # 边框粗细
    BOX_COLOR = 'black'     
    # ========================================

    # 数据清洗
    metric_cols = [
        'ctx_raw_cos', 'ctx_raw_bert',
        'ans_raw_cos', 'ans_raw_bert',
        'ctx_rerank_cos', 'ctx_rerank_bert',
        'ans_rerank_cos', 'ans_rerank_bert'
    ]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 背景设置
    sns.set_theme(style="white", context="talk", font_scale=FONT_SCALE)
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10)) 
    plt.subplots_adjust(hspace=0.5, wspace=0.3) # 稍微加大间距，防止大字体重叠
    
    colors = {
        'ctx_raw':    (126/255, 195/255, 76/255), 
        'ans_raw':    (196/255,  188/255, 219/255), 
        'ctx_rerank': (98/255,  126/255, 186/255), 
        'ans_rerank': (246/255, 189/255, 146/255)  
    }

    def draw_hist(ax, data, color, title, xlabel):
        valid = data.dropna()
        if len(valid) > 1:
            # 自动计算 X 轴范围，避免数据挤在一起
            d_min, d_max = valid.min(), valid.max()
            padding = (d_max - d_min) * 0.2 if (d_max - d_min) > 0 else 0.05
            x_low = d_min - padding
            x_high = d_max + padding
            
            # ==========================================
            # 🔥 核心修改：画出“峰值感” 🔥
            # ==========================================
            # 1. 纯密度曲线 (KDE) + 填充颜色
            # bw_adjust=0.6: 数字越小，曲线越尖锐 (峰值越明显)
            sns.kdeplot(valid, color=color, fill=True, alpha=0.3, 
                        linewidth=3, bw_adjust=0.6, ax=ax)
            
            # 2. 底部添加数据须 (Rug Plot) - 增加科研感
            sns.rugplot(valid, color=color, height=0.05, alpha=0.6, ax=ax)
            
            # 3. 均值线 (加粗高亮)
            mean_val = valid.mean()
            ax.axvline(mean_val, color='#E63946', linestyle='--', 
                       linewidth=3, alpha=0.9, label=f'Mean: {mean_val:.3f}')
            
            # 标题 & 范围
            ax.set_title(f"{title}\n(Mean: {mean_val:.4f})", 
                         fontsize=TITLE_FONT_SIZE, fontweight='bold')
            ax.set_xlim(x_low, x_high)
            
        else:
            ax.set_title(f"{title} (No Data)", fontsize=TITLE_FONT_SIZE)
            ax.set_xlim(0, 1)

        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONT_SIZE)
        
        # 刻度字体
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

        # 黑色边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BOX_COLOR)
            spine.set_linewidth(BOX_LINE_WIDTH)

    # === Row 1: Cosine Similarity ===
    draw_hist(axes[0,0], df['ctx_raw_cos'],    colors['ctx_raw'],    "1. Raw Context", "Cosine")
    draw_hist(axes[0,1], df['ans_raw_cos'],    colors['ans_raw'],    "2. Raw Answer", "Cosine")
    draw_hist(axes[0,2], df['ctx_rerank_cos'], colors['ctx_rerank'], "3. Rerank Context", "Cosine")
    draw_hist(axes[0,3], df['ans_rerank_cos'], colors['ans_rerank'], "4. Rerank Answer", "Cosine")

    # === Row 2: BERTScore ===
    draw_hist(axes[1,0], df['ctx_raw_bert'],    colors['ctx_raw'],    "1. Raw Context", "BERTScore")
    draw_hist(axes[1,1], df['ans_raw_bert'],    colors['ans_raw'],    "2. Raw Answer", "BERTScore")
    draw_hist(axes[1,2], df['ctx_rerank_bert'], colors['ctx_rerank'], "3. Rerank Context", "BERTScore")
    draw_hist(axes[1,3], df['ans_rerank_bert'], colors['ans_rerank'], "4. Rerank Answer", "BERTScore")

    fig.suptitle("RAG A/B Test: Raw vs Rerank", 
                 fontweight='bold', fontsize=MAIN_TITLE_SIZE, y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 样式调整版绘图已保存至: {save_path}")
    

# ==========================================
# 5. 主流程
# ==========================================
def main():
    mode = choose_retrieval_mode()
    POOL_K = 45
    FINAL_K = 15
    
    mode_dir = mode.capitalize()  # Hybrid / Dense / Bm25
    MODE_OUT_DIR = os.path.join(OUTPUT_DIR, mode_dir)
    os.makedirs(MODE_OUT_DIR, exist_ok=True)

    TEMP_BACKUP_PATH = os.path.join(MODE_OUT_DIR, "temp_results_backup.csv")
    FINAL_RESULT_PATH = os.path.join(MODE_OUT_DIR, "final_eval_AB_test.csv")
    DIST_PLOT_PATH = os.path.join(MODE_OUT_DIR, "score_distribution_2x4.png")

    print("⏳ 初始化向量库...")
    need_bm25 = mode in ("hybrid", "bm25")   # dense-only 不建 BM25
    try:
        load_vector_store(build_bm25=need_bm25)
    except Exception as e:
        print(f"⚠️ Warning: {e}")

    if not update_client_settings(USER_API_KEY, USER_BASE_URL):
        return

    # 加载数据
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 数据文件未找到: {TEST_DATA_PATH}")
        return
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 检查备份
    records = []
    if os.path.exists(TEMP_BACKUP_PATH):
        if input(f"❓ 发现备份，是否直接计算? (y/n): ").lower() == 'y':
            records = pd.read_csv(TEMP_BACKUP_PATH).to_dict('records')

    # 生成阶段
    if not records:
        print(f"🚀 开始 A/B Test 评估流程 (N={len(data)})...")
        print("⚠️ 注意：每条数据将调用 2 次 LLM 生成 (Raw & Rerank)，速度会较慢")
        
        for idx, item in enumerate(tqdm(data, desc="Dual Pipeline")):
            q = item.get('question', '')
            gt = item.get('ground_truth', '')
            
            try:
                # 🔥 执行双路生成
                res_dict = manual_rag_pipeline_ab_test(q, retrieval_mode=mode, pool_k=POOL_K, final_k=FINAL_K)
                
                # 清洗两个答案
                ans_raw = clean_rag_output(res_dict['ans_raw'])
                ans_rerank = clean_rag_output(res_dict['ans_rerank'])
                
            except Exception as e:
                print(f"\n[Error] {e}")
                res_dict = {"ctx_raw":"", "ctx_rerank":""}
                ans_raw, ans_rerank = "", ""

            records.append({
                "id": idx + 1,
                "question": q,
                "ground_truth": gt,
                # Path A: Raw Top-15
                "ctx_raw": res_dict.get('ctx_raw', ''),
                "ans_raw": ans_raw,
                # Path B: Rerank Top-15
                "ctx_rerank": res_dict.get('ctx_rerank', ''),
                "ans_rerank": ans_rerank
            })
            pd.DataFrame(records).to_csv(TEMP_BACKUP_PATH, index=False, encoding='utf-8-sig')

    df = pd.DataFrame(records)

    # 计算阶段
    print("\n📊 开始计算 8 组指标...")
    
    gt_list = df['ground_truth'].fillna("").astype(str).tolist()
    
    # 截断 Context 防止 BERTScore 显存溢出 (取前 2000 字符)
    raw_ctx_list = df['ctx_raw'].fillna("").astype(str).apply(lambda x: x[:2000]).tolist()
    raw_ans_list = df['ans_raw'].fillna("").astype(str).tolist()
    
    rerank_ctx_list = df['ctx_rerank'].fillna("").astype(str).apply(lambda x: x[:2000]).tolist()
    rerank_ans_list = df['ans_rerank'].fillna("").astype(str).tolist()

    # --- Group 1: Raw (No Rerank) ---
    print("\n>>> [1/4] Evaluation: Raw Context (Top-15)...")
    df['ctx_raw_cos'], df['ctx_raw_bert'] = calculate_metrics(raw_ctx_list, gt_list, "Raw_Ctx")
    
    print("\n>>> [2/4] Evaluation: Raw Answer...")
    df['ans_raw_cos'], df['ans_raw_bert'] = calculate_metrics(raw_ans_list, gt_list, "Raw_Ans")

    # --- Group 2: With Rerank ---
    print("\n>>> [3/4] Evaluation: Reranked Context (Top-15)...")
    df['ctx_rerank_cos'], df['ctx_rerank_bert'] = calculate_metrics(rerank_ctx_list, gt_list, "Rerank_Ctx")
    
    print("\n>>> [4/4] Evaluation: Reranked Answer...")
    df['ans_rerank_cos'], df['ans_rerank_bert'] = calculate_metrics(rerank_ans_list, gt_list, "Rerank_Ans")

    # 保存最终结果
    df.to_csv(FINAL_RESULT_PATH, index=False, encoding='utf-8-sig')

    # 打印对比摘要
    print("\n" + "="*80)
    print("📈 A/B Test Summary (Mean Scores)")
    print("="*80)
    summary = pd.DataFrame({
        "Metric": ["Context Cosine", "Context BERTScore", "Answer Cosine", "Answer BERTScore"],
        "Raw (Top-15)": [
            df['ctx_raw_cos'].mean(), df['ctx_raw_bert'].mean(),
            df['ans_raw_cos'].mean(), df['ans_raw_bert'].mean()
        ],
        "Rerank (Top-15)": [
            df['ctx_rerank_cos'].mean(), df['ctx_rerank_bert'].mean(),
            df['ans_rerank_cos'].mean(), df['ans_rerank_bert'].mean()
        ]
    })
    
    # 计算提升幅度
    for col in ["Raw (Top-15)", "Rerank (Top-15)"]:
        summary[col] = summary[col].round(4)
        
    print(summary.to_string(index=False))
    print("="*80)
    
    # 绘图
    try:
        plot_distributions_2x4(df, DIST_PLOT_PATH)
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")

if __name__ == "__main__":
    main()
