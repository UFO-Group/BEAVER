# rag_llm_score.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import time
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    from rich.console import Console
    _console = Console()
except Exception:
    _console = None

from Agent.Agent_Config.deepseek_client import call_deepseek_llm_Score

# ----------------------------
# JSON parsing helpers
# ----------------------------
def _extract_first_json(text: str) -> Optional[Any]:
    """
    Enhanced JSON extraction robust to DeepSeek-R1's <think> tags and markdown blocks.
    """
    if not text:
        return None

    # 1. 移除可能存在的 <think> 标签 (DeepSeek-R1 特有)
    # 这一步非常重要，防止解析到思考过程中的伪 JSON
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. 尝试寻找标准的 markdown json 代码块
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except:
            pass # 继续尝试其他方法

    # 3. 暴力寻找最外层的 [] 或 {}
    try:
        # 针对 Batch 模式 (寻找数组)
        m = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.S)
        if m: return json.loads(m.group(1))
        
        # 针对 Pointwise 模式 (寻找对象)
        m = re.search(r"(\{.*\})", text, flags=re.S)
        if m: return json.loads(m.group(1))
    except:
        pass

    return None
    
def _hash_key(query: str, passage: str) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8", errors="ignore"))
    h.update(b"\n---\n")
    h.update(passage.encode("utf-8", errors="ignore"))
    return h.hexdigest()

# ----------------------------
# Cache (optional)
# ----------------------------
def _load_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path or not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_cache(
    cache_path: Path,
    cache: Dict[str, Any],
    retries: int = 5,
    sleep_s: float = 0.25,
) -> None:
    """
    Windows-friendly atomic save:
    - 使用唯一 tmp 文件名，避免多个进程/线程抢同一个 .tmp
    - 保存前先读取磁盘上的最新版本并 merge，尽量减少并发覆盖
    - rename/replace 失败时短暂重试
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            # 先读磁盘最新缓存，再与当前 cache 合并，减少并发写丢失
            disk_cache = _load_cache(cache_path)
            merged = {}
            if isinstance(disk_cache, dict):
                merged.update(disk_cache)
            if isinstance(cache, dict):
                merged.update(cache)

            tmp = cache_path.with_name(
                f"{cache_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )

            with tmp.open("w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, cache_path)
            return

        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(sleep_s)

        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(sleep_s)

        finally:
            # 尽量清理失败残留的 tmp
            try:
                if 'tmp' in locals() and tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass


# ----------------------------
# Scoring prompts
# ----------------------------
def _build_batch_prompt(query: str, passages: List[str]) -> str:
    """
    Build a strict JSON-only batch scoring prompt.
    """
    items = []
    for i, p in enumerate(passages, start=1):
        items.append(f"[{i}] {p}")

    joined = "\n\n".join(items)

    return f"""You are a strict RAG evidence reranker for scientific QA.

Goal:
Given a Question and multiple Passages, score each passage by how directly it contains information that can answer the question.

Scoring rubric (integer):
0 = irrelevant
1 = related but no direct answer
2 = partially answers / useful facts
3 = directly answers / contains key data (numbers/conditions/explicit statement)

Output requirements:
- Return ONLY valid JSON array (no markdown, no extra text).
- The array length must equal the number of passages.
- Each element must follow this schema:
  {{
    "idx": <int, 1-based>,
    "score": <int 0-3>,
    "has_answer": <true/false>,
    "rationale": "<one short sentence>"
  }}

Question:
{query}

Passages:
{joined}
""".strip()


def _build_point_prompt(query: str, passage: str) -> str:
    return f"""You are a strict RAG evidence reranker for scientific QA.

Task:
Given a Question and a Passage, judge whether the passage contains information that can directly answer the question.

Scoring rubric (integer):
0 = irrelevant
1 = related but no direct answer
2 = partially answers / useful facts
3 = directly answers / contains key data (numbers/conditions/explicit statement)

Output requirements:
Return ONLY valid JSON:
{{
  "score": <int 0-3>,
  "has_answer": <true/false>,
  "rationale": "<one short sentence>"
}}

Question:
{query}

Passage:
{passage}
""".strip()


# ----------------------------
# Core scoring functions
# ----------------------------
def score_passages_batch(
    query: str,
    passages: List[str],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    retries: int = 1,
    sleep_s: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Score passages in a single LLM call. Returns list of dicts (idx, score, has_answer, rationale).
    If parsing fails, raises ValueError.
    """
    prompt = _build_batch_prompt(query, passages)

    last_raw = ""
    for t in range(retries + 1):
        raw = call_deepseek_llm_Score(prompt, temperature=temperature, max_tokens=max_tokens)
        last_raw = raw
        obj = _extract_first_json(raw)
        if isinstance(obj, list) and len(obj) == len(passages):
            # normalize
            out = []
            for i, it in enumerate(obj, start=1):
                score = int(it.get("score", 0)) if isinstance(it, dict) else 0
                has_answer = bool(it.get("has_answer", False)) if isinstance(it, dict) else False
                rationale = str(it.get("rationale", ""))[:240] if isinstance(it, dict) else ""
                out.append({"idx": i, "score": max(0, min(3, score)), "has_answer": has_answer, "rationale": rationale})
            return out
        time.sleep(sleep_s)

    raise ValueError(f"Batch scoring JSON parse failed. Raw:\n{last_raw[:1200]}")


def score_passage_pointwise(
    query: str,
    passage: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    retries: int = 1,
    sleep_s: float = 0.2,
) -> Dict[str, Any]:
    """
    Score a single passage. Always returns a dict with score/has_answer/rationale (fallback safe).
    """
    prompt = _build_point_prompt(query, passage)

    last_raw = ""
    for _ in range(retries + 1):
        raw = call_deepseek_llm_Score(prompt, temperature=temperature, max_tokens=max_tokens)
        last_raw = raw
        obj = _extract_first_json(raw)
        if isinstance(obj, dict):
            score = int(obj.get("score", 0))
            score = max(0, min(3, score))
            return {
                "score": score,
                "has_answer": bool(obj.get("has_answer", False)),
                "rationale": str(obj.get("rationale", ""))[:240],
            }
        time.sleep(sleep_s)

    # fallback
    return {"score": 0, "has_answer": False, "rationale": "parse_failed"}


# ----------------------------
# Public API: rerank results
# ----------------------------
def rerank_with_llm_score(
    query: str,
    results: List[Dict[str, Any]],
    top_n: int = 5,
    batch_size: int = 10,
    max_chars: int = 1200,
    use_batch: bool = True,
    cache_path: Optional[str] = None,
    fallback_to_cosine: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Rerank retrieval results using call_deepseek_llm_Score.
    Returns: (top_results, scored_all)

    - Uses batch scoring by default (fewer API calls).
    - Optional disk cache to reuse scoring results.
    - Fallback: if scoring fails, returns cosine-top_n if fallback_to_cosine=True.
    """
    if not results:
        return [], []

    # Prepare cache
    cache_file = Path(cache_path) if cache_path else None
    cache = _load_cache(cache_file) if cache_file else {}

    # Prepare passages
    passages = []
    for r in results:
        txt = str(r.get("evidence", "") or "")
        passages.append(txt[:max_chars])

    scored_meta: List[Dict[str, Any]] = []
    try:
        if use_batch:
            # Score in batches
            for start in range(0, len(passages), batch_size):
                chunk = passages[start : start + batch_size]

                # Try cache hits first
                chunk_scores: List[Optional[Dict[str, Any]]] = [None] * len(chunk)
                miss_indices = []
                miss_passages = []

                for j, p in enumerate(chunk):
                    key = _hash_key(query, p)
                    if key in cache:
                        chunk_scores[j] = cache[key]
                    else:
                        miss_indices.append(j)
                        miss_passages.append(p)

                # Call LLM for misses
                if miss_passages:
                    batch_out = score_passages_batch(query, miss_passages)
                    for k, one in enumerate(batch_out):
                        j = miss_indices[k]
                        chunk_scores[j] = {
                            "score": int(one["score"]),
                            "has_answer": bool(one["has_answer"]),
                            "rationale": str(one["rationale"])[:240],
                        }
                        key = _hash_key(query, miss_passages[k])
                        cache[key] = chunk_scores[j]

                # Merge
                for j, sc in enumerate(chunk_scores):
                    # should never be None now, but keep safe
                    sc = sc or {"score": 0, "has_answer": False, "rationale": "missing"}
                    scored_meta.append(sc)
        else:
            # Pointwise scoring
            for p in passages:
                key = _hash_key(query, p)
                if key in cache:
                    sc = cache[key]
                else:
                    sc = score_passage_pointwise(query, p)
                    cache[key] = sc
                scored_meta.append(sc)

        # Save cache
        if cache_file:
            try:
                _save_cache(cache_file, cache)
            except Exception as e:
                if _console:
                    _console.print(f"[yellow]⚠️ Cache save skipped: {e}[/yellow]")

        # Attach scores back to results
        scored_all = []
        for r, sc in zip(results, scored_meta):
            r2 = dict(r)
            r2["rerank_score"] = int(sc.get("score", 0))
            r2["rerank_has_answer"] = bool(sc.get("has_answer", False))
            r2["rerank_rationale"] = str(sc.get("rationale", ""))[:240]
            scored_all.append(r2)

        # Sort: rerank_score desc, then cosine score desc (if exists)
        scored_all.sort(
            key=lambda x: (x.get("rerank_score", 0), x.get("score", 0.0)),
            reverse=True
        )

        return scored_all[:top_n], scored_all

    except Exception as e:
        if _console:
            _console.print(f"[yellow]⚠️ Rerank failed, fallback. Reason: {e}[/yellow]")
        if not fallback_to_cosine:
            raise

        # Fallback to cosine top_n (or original order if cosine missing)
        fallback = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        return fallback[:top_n], fallback


# ----------------------------
# Optional CLI for quick testing
# ----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LLM Score Reranker for RAG results")
    ap.add_argument("--query", required=True, help="Question/query text")
    ap.add_argument("--in_json", required=True, help="Input JSON file of retrieve results (list)")
    ap.add_argument("--out_json", default="reranked_results.json", help="Output JSON file")
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--no_batch", action="store_true")
    ap.add_argument("--cache", default="", help="Cache file path (optional)")
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    top, all_scored = rerank_with_llm_score(
        query=args.query,
        results=results,
        top_n=args.top_n,
        batch_size=args.batch_size,
        use_batch=(not args.no_batch),
        cache_path=(args.cache or None),
    )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"top": top, "all": all_scored}, f, ensure_ascii=False, indent=2)

    if _console:
        _console.print(f"[green]✅ Saved reranked results to: {args.out_json}[/green]")
    else:
        print(f"Saved reranked results to: {args.out_json}")
