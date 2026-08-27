import os
import re
import textwrap
import numpy as np
import pandas as pd
from rich.table import Table
from rich.console import Console

# ==========================================
# 📦 Local Imports
# ==========================================
from .rag_utils import rewrite_query_with_domain, get_source_id
from . import vector_store
from .vector_store import load_vector_store, tokenize_en
from .rag_llm_score import rerank_with_llm_score

import Agent.Agent_Config.agent_config as config
from Agent.Agent_Config.deepseek_client import call_deepseek_llm, get_embedding_via_api
from Agent.Utils.file_utils import save_step_result 

# Initialize console
console = Console()

RAG_RETRIEVE_TOP_K = int(os.getenv("RAG_RETRIEVE_TOP_K", 45))
RAG_RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", 15))
RAG_RERANK_BATCH_SIZE = int(os.getenv("RAG_RERANK_BATCH_SIZE", 15))

# float32 + 只归一化一次 + 点积=cosine
def _ensure_dense_ready():
    X = vector_store.embeddings

    # ✅ 强烈建议：在 load_vector_store 阶段就确保 float32
    # 这里不强转，避免运行时复制整矩阵导致内存峰值
    # 但至少缓存 norms（不修改 X 本体）
    if not hasattr(vector_store, "emb_norms") or vector_store.emb_norms is None or len(vector_store.emb_norms) != X.shape[0]:
        # ✅ einsum 不会产生 X*X 的大临时矩阵
        row_sq = np.einsum("ij,ij->i", X, X)
        vector_store.emb_norms = (np.sqrt(row_sq).astype(np.float32) + 1e-12)  # shape=(N,)

def _dense_topk(query: str, top_k: int):
    _ensure_dense_ready()
    X = vector_store.embeddings
    Xn = vector_store.emb_norms  # (N,)

    q = get_embedding_via_api(query).astype(np.float32, copy=False).reshape(-1)
    qn = np.float32(np.sqrt(np.sum(q * q, dtype=np.float32)) + 1e-12)
    q /= qn  # 让 q 变成单位向量

    # 点积
    scores = X @ q

    # ✅ 如果 X 是 float64，这里 scores 可能是 float64；可以显式转成 float32 省点内存
    scores = scores.astype(np.float32, copy=False)

    # 余弦归一化（只除以 X 的 norm，因为 q 已经单位化）
    scores /= Xn

    idx = np.argpartition(scores, -top_k)[-top_k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx, scores


from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

# 可选：限制 BLAS 线程，避免“线程爆炸”
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

def _topk_one_shard(shard: dict, q: np.ndarray, qn: float, top_k: int):
    X = shard["emb"]      # (n,d) memmap
    Xn = shard["norms"]   # (n,)  memmap

    ctx = threadpool_limits(limits=1) if threadpool_limits else nullcontext()
    with ctx:
        # 精确全扫这个 shard
        dots = X @ q  # (n,)
        dots = dots.astype(np.float32, copy=False)
        scores = dots / (Xn * qn)

    # shard 内精确 top_k
    idx = np.argpartition(scores, -top_k)[-top_k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    # 转全局索引
    gidx = idx + shard["offset"]
    return gidx.astype(np.int64, copy=False), scores[idx]

def dense_topk_exact_sharded(query: str, top_k: int = 45, n_workers: int | None = None):
    # 确保 shard 已加载
    if not getattr(vector_store, "shards", None):
        load_vector_store(build_bm25=False)

    q = get_embedding_via_api(query).astype(np.float32, copy=False).reshape(-1)
    qn = float(np.linalg.norm(q) + 1e-12)

    # ✅ 精确全局 top_k：每个 shard 取 top_k 再全局 merge（数学上不会漏）
    workers = n_workers or min(len(vector_store.shards), (os.cpu_count() or 8))
    all_idx = []
    all_sc = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_topk_one_shard, shard, q, qn, top_k) for shard in vector_store.shards]
        for fu in as_completed(futs):
            idx, sc = fu.result()
            all_idx.append(idx)
            all_sc.append(sc)

    all_idx = np.concatenate(all_idx)
    all_sc = np.concatenate(all_sc)

    best = np.argpartition(all_sc, -top_k)[-top_k:]
    best = best[np.argsort(all_sc[best])[::-1]]

    return all_idx[best], all_sc[best]
    
# ============ Text Cleaning ============
def clean_text(text: str) -> str:
    """清理文本中的乱码和多余空格"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('–', '-').replace('—', '-')
    return text.strip()
# ============ 🔥 [新增] RRF 融合算法 ============
def weighted_reciprocal_rank(list1, list2, k=60):
    """
    Reciprocal Rank Fusion (RRF): 合并两个排名列表。
    list1, list2 格式: [{'index_id': 123, 'score': 0.9}, ...]
    """
    fused_scores = {}
    
    # 处理列表 1 (Vector)
    for rank, item in enumerate(list1):
        idx = item['index_id']
        # RRF 公式: 1 / (k + rank)
        fused_scores[idx] = fused_scores.get(idx, 0) + (1 / (k + rank + 1))
        
    # 处理列表 2 (BM25)
    for rank, item in enumerate(list2):
        idx = item['index_id']
        fused_scores[idx] = fused_scores.get(idx, 0) + (1 / (k + rank + 1))
    
    # 按融合分数排序，返回 [(index_id, final_score), ...]
    sorted_indices = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_indices

# ============ 🔥 [新增] BM25 检索函数 ============   
def retrieve_bm25(query: str, top_k: int):
    if vector_store.bm25_model is None:
        return []

    tokenized_query = tokenize_en(query)
    doc_scores = vector_store.bm25_model.get_scores(tokenized_query)

    top_idx = np.argsort(doc_scores)[::-1][:top_k]
    results = []
    for i in top_idx:
        score = doc_scores[i]
        if score > 0:
            results.append({"index_id": i, "score": float(score)})
    return results

DEFAULT_RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "hybrid").lower()
# ============ [修改] 主检索函数 ============
def retrieve_evidence(query: str, top_k: int = 50, retrieval_mode: str = "hybrid"):  # 建议默认 top_k 调大
    """
    Hybrid Retrieval: Vector Search + BM25 -> RRF Fusion

    retrieval_mode:
      - "hybrid": Vector + BM25 -> RRF (默认，保持你现有逻辑不变)
      - "dense" : Vector only
      - "bm25"  : BM25 only
    """
    mode = (retrieval_mode or "hybrid").lower()
    if mode not in ("hybrid", "dense", "bm25"):
        mode = "hybrid"

    need_dense = mode in ("hybrid", "dense")
    need_bm25  = mode in ("hybrid", "bm25")

    # Check loading（只在需要时要求对应对象已加载）
    need_shards = need_dense and (not getattr(vector_store, "shards", None))
    
    if (
        vector_store.metadata is None
        or need_shards
        or (need_bm25 and vector_store.bm25_model is None)
    ):
        console.print("[yellow]⚠️ Metadata/BM25 not loaded, auto-loading...[/yellow]")
        load_vector_store(build_bm25=need_bm25)

    # -------------------------
    # 1) Dense-only
    # -------------------------
    if mode == "dense":
        vec_top_idx, vec_top_scores = dense_topk_exact_sharded(query, top_k=top_k)

        vec_results = [{"index_id": int(i), "score": float(s)} for i, s in zip(vec_top_idx, vec_top_scores)]
    
        final_results = []
        for rank, (idx, s) in enumerate(zip(vec_top_idx, vec_top_scores), start=1):
            row = vector_store.metadata.iloc[int(idx)]

            filename = row.get("filename", "Unknown_File.txt")
            raw_text = row.get("text", "")
            topic = row.get("auto_topic", "")
            clean_txt = clean_text(str(raw_text))

            final_results.append({
                "rank": rank,
                "score": round(float(s), 4),  # 余弦相似度
                "source": filename,
                "filename": filename,
                "topic": topic,
                "evidence": clean_txt
            })
        return final_results

    # -------------------------
    # 2) BM25-only
    # -------------------------
    if mode == "bm25":
        bm25_results = retrieve_bm25(query, top_k=top_k)  # 期望返回 [{"index_id":..., "score":...}, ...]

        final_results = []
        for rank, item in enumerate(bm25_results[:top_k], start=1):
            idx = item["index_id"]
            row = vector_store.metadata.iloc[idx]

            filename = row.get("filename", "Unknown_File.txt")
            raw_text = row.get("text", "")
            topic = row.get("auto_topic", "")
            clean_txt = clean_text(str(raw_text))

            final_results.append({
                "rank": rank,
                "score": round(float(item["score"]), 4),  # BM25 分数
                "source": filename,
                "filename": filename,
                "topic": topic,
                "evidence": clean_txt
            })
        return final_results

    # -------------------------
    # 3) Hybrid（原逻辑：Vector + BM25 -> RRF）
    # -------------------------

    # --- 1. Vector Search (Semantic) ---
    vec_top_idx, vec_top_scores = dense_topk_exact_sharded(query, top_k=top_k)
    vec_results = [{"index_id": int(i), "score": float(s)} for i, s in zip(vec_top_idx, vec_top_scores)]

    # --- 2. BM25 Search (Keyword) ---
    bm25_results = retrieve_bm25(query, top_k=top_k)

    # --- 3. Fusion (RRF) ---
    # 合并两路结果
    fused_list = weighted_reciprocal_rank(vec_results, bm25_results, k=60)

    # --- 4. Format Results ---
    final_results = []
    # 只取融合后的前 top_k 个
    for rank, (idx, rrf_score) in enumerate(fused_list[:top_k], start=1):
        row = vector_store.metadata.iloc[idx]

        filename = row.get("filename", "Unknown_File.txt")
        raw_text = row.get("text", "")
        topic = row.get("auto_topic", "")
        clean_txt = clean_text(str(raw_text))

        final_results.append({
            "rank": rank,
            "score": round(float(rrf_score), 4),  # 注意：这里是 RRF 分数，不是余弦相似度了
            "source": filename,
            "filename": filename,
            "topic": topic,
            "evidence": clean_txt
        })

    return final_results

def retrieve_and_rerank_evidence(
    query: str,
    domain: dict | None = None,
    top_k: int = 15,
    query_rewrite_mode: str = "domain_append",
    export_csv: bool = True,
    step_id: str | None = None,
    export_path: str | None = None,
    retrieval_mode: str | None = None,
    use_rerank: bool = True,
    retrieve_top_k: int | None = None,
    rerank_top_n: int | None = None,
    rerank_batch_size: int | None = None,
    rerank_cache_path: str | None = None,
):
    retrieve_top_k = retrieve_top_k or RAG_RETRIEVE_TOP_K
    rerank_top_n = rerank_top_n or RAG_RERANK_TOP_K
    rerank_batch_size = rerank_batch_size or RAG_RERANK_BATCH_SIZE
    """
    Retrieval-only function:
    Retrieve + (Optional) Rerank
    NO answer generation here.
    """

    # --- Step 0: Domain-aware query rewriting ---
    if domain is not None:
        enhanced_query = rewrite_query_with_domain(query, domain, mode=query_rewrite_mode)
    else:
        enhanced_query = query

    mode = (retrieval_mode or DEFAULT_RETRIEVAL_MODE).lower()

    # --- Step 1: Retrieval ---
    if use_rerank:
        console.print(
            f"\n🔍 [bold cyan]Retrieval Only (candidate pool)[/bold cyan]\n"
            f"   Base query      : {query}\n"
            f"   Enhanced query  : {enhanced_query}\n"
            f"   retrieve_top_k  : {retrieve_top_k}\n"
            f"   rerank_top_n    : {rerank_top_n}\n"
        )
        results = retrieve_evidence(enhanced_query, top_k=retrieve_top_k, retrieval_mode=mode)
    else:
        console.print(
            f"\n🔍 [bold cyan]Retrieval Only (no rerank)[/bold cyan]\n"
            f"   Base query      : {query}\n"
            f"   Enhanced query  : {enhanced_query}\n"
            f"   top_k (final)   : {top_k}\n"
        )
        results = retrieve_evidence(enhanced_query, top_k=top_k, retrieval_mode=mode)

    scored_all = None

    # --- Build rerank cache path ---
    effective_rerank_cache_path = rerank_cache_path

    if use_rerank and not effective_rerank_cache_path:
        if export_path:
            cache_dir = os.path.join(export_path, ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f"{step_id or 'retrieval'}_rerank_cache.json"
            effective_rerank_cache_path = os.path.join(cache_dir, cache_name)
        else:
            # 如果没有 export_path，就退回到全局缓存目录，但文件名带 step_id，降低冲突
            os.makedirs("Agent/RAG/.cache", exist_ok=True)
            cache_name = f"{step_id or 'retrieval'}_rerank_cache.json"
            effective_rerank_cache_path = os.path.join("Agent/RAG/.cache", cache_name)
    
    # --- Step 1.5: Rerank ---
    if use_rerank and results:
        console.print(f"\n🧪 [bold cyan]Reranking {len(results)} passages via Score model...[/bold cyan]")
        topN, scored_all = rerank_with_llm_score(
            query=query,
            results=results,
            top_n=rerank_top_n,
            batch_size=rerank_batch_size,
            cache_path=effective_rerank_cache_path,
        )
        results = topN
        console.print(f"[green]✔ Rerank done. Keeping Top {len(results)} passages.[/green]")
        print("[DEBUG] A: rerank returned", flush=True)
    
    # --- Step 2: Save retrieval artifacts only ---
    if export_csv:
        tag = step_id or "retrieval"
        print("[DEBUG] B: before save retrieval_only", flush=True)
    
        try:
            save_step_result(
                step_id=tag,
                step_type="retrieval_only",
                content=results,
                step_log_dir=export_path,
                console=None,   # 先不要把 rich/streamlit console 传进去
            )
            print("[DEBUG] C: after save retrieval_only", flush=True)
        except Exception as e:
            print(f"[DEBUG] C_ERR: save retrieval_only failed: {e}", flush=True)
            raise
    
        if use_rerank and scored_all is not None:
            print("[DEBUG] D: before save rerank_scored_all", flush=True)
            try:
                save_step_result(
                    step_id=tag,
                    step_type="rerank_scored_all",
                    content=scored_all,
                    step_log_dir=export_path,
                    console=None,
                )
                print("[DEBUG] E: after save rerank_scored_all", flush=True)
            except Exception as e:
                print(f"[DEBUG] E_ERR: save rerank_scored_all failed: {e}", flush=True)
                raise
    
    print("[DEBUG] F: before return retrieval_pack", flush=True)
    
    # 🚀 补上缺失的 return 语句！
    return {
        "enhanced_query": enhanced_query,
        "retrieval_mode": mode,
        "evidence_items": results,
        "scored_all": scored_all
    }

# ============ RAG Main Function ============
def rag_answer(
    query: str,
    domain: dict | None = None,
    top_k: int = 15,
    query_rewrite_mode: str = "domain_append",          # 若Rerank关闭，则应该与rerank_top_n相似     
    export_csv: bool = True,
    step_id: str | None = None,
    # 🔥 [新增参数] 允许外部传入保存路径
    export_path: str | None = None, 
    
    # ✅ 新增：检索模式开关
    retrieval_mode: str | None = None,

    # Rerank params
    use_rerank: bool = True,
    retrieve_top_k: int | None = None,
    rerank_top_n: int | None = None,
    rerank_batch_size: int | None = None,
    rerank_cache_path: str | None = None,
):
    """
    Main RAG function: Retrieve + (Optional) Rerank + Generate Answer
    """

    retrieval_pack = retrieve_and_rerank_evidence(
        query=query,
        domain=domain,
        top_k=top_k,
        query_rewrite_mode=query_rewrite_mode,
        export_csv=False,   # 这里不要重复保存 retrieval_only
        step_id=step_id,
        export_path=export_path,
        retrieval_mode=retrieval_mode,
        use_rerank=use_rerank,
        retrieve_top_k=retrieve_top_k,
        rerank_top_n=rerank_top_n,
        rerank_batch_size=rerank_batch_size,
        rerank_cache_path=rerank_cache_path,
    )

    enhanced_query = retrieval_pack["enhanced_query"]
    mode = retrieval_pack["retrieval_mode"]
    results = retrieval_pack["evidence_items"]
    scored_all = retrieval_pack["scored_all"]

    # --- Step 2: Display Results (Console) ---
    table = Table(title="📑 Retrieved Evidence Passages")
    table.add_column("Rank", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Topic", style="magenta")
    table.add_column("Source File", overflow="fold", style="cyan")
    table.add_column("Evidence (Preview)", overflow="fold", max_width=120)

    for idx, r in enumerate(results, start=1):
        preview = textwrap.fill(r.get("evidence", ""), width=110)
        score_val = r.get("rerank_score") if use_rerank else r.get("score")
        # 确保显示的是 filename
        filename = r.get("filename", r.get("source", "Unknown"))
        topic = r.get("topic", "N/A")
        table.add_row(
            str(idx),                          
            str(score_val),
            str(topic),
            str(filename), 
            preview
        )
    console.print(table)

    # --- Step 3: Build Prompt ---
    context_pieces = []
    for i, r in enumerate(results, start=1):
        content = r.get('evidence', '').strip()
        topic = r.get('topic', '') # 获取标签
        
        # 🔥 修改 2：在 Prompt 上下文中显式加入 Topic 信息
        # 这样大模型在阅读 Evidence [1] 时，能直接看到 "Topic: Lithium-ion battery"
        if topic:
            filename = r.get("filename", "Unknown")
            context_entry = f"Evidence [{i}]:\nSource: {filename}\nTopic: {topic}\nContent: {content}"
        else:
            context_entry = f"Evidence [{i}]:\nContent: {content}"
            
        context_pieces.append(context_entry)

    context_str = "\n\n".join(context_pieces)

    prompt = f"""You are an expert polymer scientist.
Below are extracted text segments from research papers.

Context Evidence:
{context_str}

Question: {query}

### INSTRUCTIONS:
1. Base your core scientific arguments heavily on the provided Context Evidence.
2. **CITATION RULE**: When you use information from a specific evidence, cite it using the format [index]. For example: "Polymer X degrades rapidly [1]".
3. **STRICTLY FORBIDDEN**: DO NOT generate a "References" list at the end.

Please provide a scientifically rigorous, detailed, and comprehensive answer in English.
"""

    # --- Step 4: Call LLM ---
    console.print("\n🧠 [bold green]Generating answer with DeepSeek...[/bold green]")
    
    # 1. 获取原始响应
    raw_response = call_deepseek_llm(prompt)

    # 2. 🔥【新增】清洗 <think> 标签逻辑
    # 使用非贪婪匹配移除 <think>...</think> 及其内容
    cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    # 3. 兜底逻辑：防止清洗后为空
    if not cleaned_answer:
        console.print("[yellow]⚠️ Warning: Response became empty after cleaning <think> tags. Reverting to raw response.[/yellow]")
        cleaned_answer = raw_response

    # Step 5: build refs only for logging / standalone export
    ref_section = "\n\n### References\n"
    for i, r in enumerate(results, start=1):
        ref_section += f"[{i}] {get_source_id(r)}\n"
    
    final_output_for_log = cleaned_answer + ref_section
    
    # --- Step 6: Output Result (Console) ---
    console.rule("[bold yellow]📤 Final Answer[/bold yellow]")
    console.print(textwrap.fill(final_output_for_log, width=120))
    console.rule()

    # --- Step 7: Export Results ---
    if export_csv:
        tag = step_id or "rag"
        
        # 1. Save retrieval details (CSV)
        save_step_result(
            step_id=tag,
            step_type="retrieval_with_answer",
            content=results,
            step_log_dir=export_path,  # ✅ 修改：传入路径
            console=console
        )

        # 2. Save the final Answer (TXT)
        txt_content = f"Question: {query}\n"
        txt_content += f"Enhanced query: {enhanced_query}\n\n"
        txt_content += "Answer:\n"
        txt_content += f"{final_output_for_log}\n\n" 
        
        txt_content += "--- Evidence Details ---\n"
        for i, r in enumerate(results, start=1):
            txt_content += f"[{i}] {r.get('filename')}\n"
            txt_content += f"    Score: {r.get('rerank_score', r.get('score'))}\n"
            txt_content += f"    Content: {r.get('evidence','')}\n\n"

        save_step_result(
            step_id=tag,
            step_type="rag_answer",
            content=txt_content,
            step_log_dir=export_path,  # ✅ 修改：传入路径
            console=console
        )

        # 3. Save Rerank Scored Full List (CSV)
        if use_rerank and scored_all is not None:
            save_step_result(
                step_id=tag,
                step_type="rerank_scored_all",
                content=scored_all,
                step_log_dir=export_path,  # ✅ 修改：传入路径
                console=console
            )

    return {
        "answer": final_output_for_log,
        "evidence_items": results,
        "query": query,
        "retrieval_mode": retrieval_mode,
    }