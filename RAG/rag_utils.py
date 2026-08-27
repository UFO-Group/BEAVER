# rag_utils.py
from typing import Dict
import re

def clean_source_id(raw: str) -> str:
    raw = str(raw).strip()
    raw = re.sub(r"_段\d+(?:-\d+)?", "", raw)
    raw = re.sub(r"\.txt$", "", raw)
    raw = re.sub(r"_\d+\.npy$", "", raw)
    return raw

def get_source_id(item: dict) -> str:
    return (
        item.get("source_id_clean")
        or item.get("chunk_file_id_raw")
        or clean_source_id(item.get("filename", "Unknown"))
    )
def normalize_text(s: str) -> str:
    """
    统一文本格式：去掉弯引号、首尾空白等。
    """
    if not isinstance(s, str):
        return ""
    s = s.replace("’", "'").strip()
    return s


def rewrite_query_with_domain(query: str, domain: dict | None, mode: str = "domain_append") -> str:
    """
    Query rewrite policy used by retrieval.

    mode:
      - "none": preserve the original query exactly.
      - "domain_append": append selected structured-domain terms when they are not already present.
    """
    base = normalize_text(query)
    mode = (mode or "domain_append").strip().lower()

    if mode == "none" or not domain:
        return base

    # Fallback to domain_append for unknown modes.
    if mode != "domain_append":
        mode = "domain_append"

    extra_parts = []
    for key in ["material_family", "modification_type", "target_property", "mechanism_type"]:
        v = normalize_text(domain.get(key, ""))
        if v and v.lower() != "unknown":
            extra_parts.append(v)

    extra_unique = []
    base_lower = base.lower()
    for t in extra_parts:
        if t.lower() not in base_lower:
            extra_unique.append(t)

    if not extra_unique:
        return base

    return base + " " + " ".join(extra_unique)

    if not domain:
        return base

    # ① 提取结构化关键词
    extra_parts = []
    for key in ["material_family", "modification_type", "target_property", "mechanism_type"]:
        v = normalize_text(domain.get(key, ""))
        if v and v.lower() != "unknown":
            extra_parts.append(v)

    # ② 去重：base query 已经包含的关键词不重复加入
    extra_unique = []
    base_lower = base.lower()
    for t in extra_parts:
        if t.lower() not in base_lower:
            extra_unique.append(t)

    # ③ 如果没有新关键词，就保持原样
    if not extra_unique:
        return base

    # ④ 拼接增强后的 query
    return base + " " + " ".join(extra_unique)

