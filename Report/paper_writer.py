import sys
import os
import re
import json
from typing import Any, Callable

try:
    from Agent.Planner.domain_router import coarsen_domain_to_report_hints
except Exception:
    try:
        from domain_router import coarsen_domain_to_report_hints
    except Exception:
        coarsen_domain_to_report_hints = None

# === 路径修正：确保能找到 Agent 包 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. 确保导入路径正确
try:
    from Agent.Agent_Config.deepseek_client import call_deepseek_llm_Report
except ImportError:
    def call_deepseek_llm_Report(p, temperature=0.0):
        return "LLM fallback content"


def strip_think_tags(text: str) -> str:
    """移除模型思维标签，保持正文稳定。"""
    if not isinstance(text, str):
        return str(text)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text.strip()

def extract_machine_readable_score_block(text: str) -> dict | None:
    """
    从正文中提取 Machine-Readable Score Block 里的 JSON。
    仅提取，不改正文。
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # fenced json
    m = re.search(
        r"(?is)##\s*\d+\.\s*Machine-Readable Score Block\s*```json\s*(\{.*?\})\s*```",
        text,
    )
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # unfenced json
    m = re.search(
        r"(?is)##\s*\d+\.\s*Machine-Readable Score Block\s*(\{.*?\})\s*(?=##\s*\d+\.|\Z)",
        text,
    )
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def remove_machine_readable_score_block(text: str) -> str:
    """
    从最终论文正文中移除 Machine-Readable Score Block 章节，
    保留其余章节与引用。
    """
    if not isinstance(text, str) or not text.strip():
        return text

    cleaned = re.sub(
        r"(?is)\n*##\s*\d+\.\s*Machine-Readable Score Block\s*.*?(?=\n##\s*\d+\.|\Z)",
        "\n",
        text,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def normalize_citations_and_rebuild_references(
    text: str,
    ordered_source_ids: list[str],
    default_heading: str = "7. References",
) -> str:
    """
    根据正文首次出现的 [n] 顺序，强制重建 References。
    这里只信 ordered_source_ids，不信模型自己写的参考文献区。
    """
    if not text or not isinstance(text, str):
        return text

    if not ordered_source_ids:
        return text

    ref_heading_match = re.search(
        r"(?im)^\s{0,3}(?:#+\s*)?(?:\d+\.\s*)?\*{0,2}References\*{0,2}\s*:?\s*$",
        text,
    )

    if ref_heading_match:
        body = text[: ref_heading_match.start()].rstrip()
        heading_line = ref_heading_match.group(0).strip()
    else:
        body = text.rstrip()
        heading_line = default_heading

    old_ref_map = {idx: sid for idx, sid in enumerate(ordered_source_ids, start=1) if sid}
    if not old_ref_map:
        return text

    cite_pat = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    first_appearance = []
    seen = set()
    for match in cite_pat.finditer(body):
        nums = [int(x.strip()) for x in match.group(1).split(",")]
        for n in nums:
            if n in old_ref_map and n not in seen:
                first_appearance.append(n)
                seen.add(n)

    if not first_appearance:
        if ref_heading_match:
            return body.rstrip() + "\n\n" + heading_line + "\n- No cited literature sources identified."
        return text

    old_to_new = {old: new for new, old in enumerate(first_appearance, start=1)}

    def _replace(match: re.Match) -> str:
        old_nums = [int(x.strip()) for x in match.group(1).split(",")]
        mapped = []
        for n in old_nums:
            if n in old_to_new:
                mapped.append(old_to_new[n])
        if not mapped:
            return match.group(0)
        mapped = sorted(set(mapped))
        return "[" + ",".join(str(x) for x in mapped) + "]"

    new_body = cite_pat.sub(_replace, body)

    new_ref_lines = [heading_line]
    for old_id in first_appearance:
        new_id = old_to_new[old_id]
        source_id = old_ref_map[old_id]
        new_ref_lines.append(f"- [{new_id}] {source_id}")

    return new_body.rstrip() + "\n\n" + "\n".join(new_ref_lines)


def _call_llm_with_fallback(
    llm_callable: Callable[..., str],
    prompt: str,
    temperature: float | None = None,
) -> str:
    """兼容不同 DeepSeek wrapper 的签名。"""
    if temperature is not None:
        try:
            return llm_callable(prompt, temperature=temperature)
        except TypeError:
            pass
    return llm_callable(prompt)


def _looks_like_final_report(text: str) -> bool:
    if not isinstance(text, str):
        return False

    has_title = bool(re.search(r"(?m)^\s*#\s+\S+", text))
    has_abstract = bool(re.search(r"(?im)^\s*##\s*Abstract\b", text))
    has_main_section = bool(
        re.search(r"(?im)^\s*##\s*(?:1\.\s*)?Introduction\b", text)
        or re.search(r"(?im)^\s*##\s*(?:2\.\s*)?(?:Mechanistic Basis|Mechanistic Rationale|Design Hypothesis|Theoretical Mechanism)\b", text)
        or re.search(r"(?im)^\s*##\s*(?:3\.\s*)?(?:Materials and Methods|Experimental Methods|Methodology)\b", text)
        or re.search(r"(?im)^\s*##\s*(?:4\.\s*)?(?:Results and Evidence-Based Discussion|Discussion|Experimental Translation Framework)\b", text)
        or re.search(r"(?im)^\s*##\s*(?:5\.\s*)?Conclusion\b", text)
        or re.search(r"(?im)^\s*##\s*(?:6\.\s*)?Conclusion\b", text)
    )

    return has_title and has_abstract and has_main_section

def _extract_best_effort_source_id(item: Any, idx: int) -> str:
    """
    从 legacy raw_evidence 中尽力提取 source id。
    提不出来时才回退到 Unknown_Source_n。
    """
    fallback = f"Unknown_Source_{idx}"

    if isinstance(item, dict):
        candidate_keys = [
            "source_id",
            "clean_id",
            "paper_id",
            "doc_id",
            "document_id",
            "id",
            "title",
        ]
        for key in candidate_keys:
            value = item.get(key)
            if value is not None:
                value = str(value).strip()
                if value:
                    return value

    text = str(item or "").strip()
    if not text:
        return fallback

    # 常见模式 1: Source ID: xxx
    m = re.search(r"(?i)source\s*id\s*:\s*([^\n\r;]+)", text)
    if m:
        val = m.group(1).strip()
        if val:
            return val[:200]

    # 常见模式 2: [ID] xxx
    m = re.search(r"(?i)\b(?:clean[_\s-]?id|paper[_\s-]?id|doc[_\s-]?id|id)\b\s*[:=]\s*([^\n\r;]+)", text)
    if m:
        val = m.group(1).strip()
        if val:
            return val[:200]

    return fallback

def _build_fallback_structured_evidence_from_raw(raw_evidence: list[Any]):
    """
    当旧 reporter 只有 raw_evidence 字符串时，尽可能构建一个最弱版本的 structured evidence。
    不强造 Pure Clean ID；只保留 Unknown_Source_n 以维持格式稳定。
    """
    blocks = []
    ref_lines = []
    ordered_source_ids = []

    if not isinstance(raw_evidence, list):
        raw_evidence = []

    for idx, item in enumerate(raw_evidence[:15], start=1):
        source_id = _extract_best_effort_source_id(item, idx)
        text = str(item or "").strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)[:1200]
        ordered_source_ids.append(source_id)
        ref_lines.append(f"[{idx}] {source_id}")
        blocks.append(
            f"Evidence [{idx}]:\n"
            f"Source ID: {source_id}\n"
            f"From step(s): legacy_raw_evidence\n"
            f"Snippet 1: {text}"
        )

    reference_map_text = "\n".join(ref_lines) if ref_lines else "[1] Unknown_Source"
    structured_evidence_text = "\n\n".join(blocks) if blocks else "No structured evidence available."
    return structured_evidence_text, reference_map_text, ordered_source_ids


CATEGORY_DISPLAY_NAMES = {
    "mechanical_design": "力学性能设计",
    "thermal_design": "热学性能设计",
    "degradation_regulation": "降解行为与结构调控",
}

CATEGORY_KEYWORDS = {
    "mechanical_design": {
        "mechanical": 3, "strength": 3, "stiffness": 3, "modulus": 4, "young's modulus": 4,
        "tensile": 3, "compression": 3, "compressive": 3, "flexural": 2, "toughness": 4,
        "ductility": 3, "elongation": 3, "fatigue": 2, "load-bearing": 4, "load bearing": 4,
        "support": 2, "fracture": 2, "failure mode": 2, "wet-state": 3, "wet state": 3,
        "mechanics": 3, "reinforcement": 2, "strength retention": 3,
        "力学": 4, "强度": 4, "模量": 4, "刚度": 4, "韧性": 4, "承载": 4, "压缩": 3, "拉伸": 3,
    },
    "thermal_design": {
        "thermal": 3, "glass transition": 5, "tg": 4, "melting": 4, "tm": 4,
        "thermal stability": 4, "heat resistance": 3, "processing window": 4, "melt processing": 4,
        "crystallization": 3, "crystallinity": 3, "dsc": 3, "dma": 2, "annealing": 2,
        "cold crystallization": 3, "heat distortion": 2, "thermomechanical": 2,
        "热学": 4, "玻璃化转变": 5, "玻璃化温度": 5, "熔点": 4, "结晶": 3, "热稳定": 4,
        "加工窗口": 4, "热变形": 2,
    },
    "degradation_regulation": {
        "degradation": 5, "degrade": 4, "hydrolysis": 5, "erosion": 4, "mass loss": 4,
        "property retention": 3, "autocatalysis": 5, "acidic byproduct": 5, "acidic": 2,
        "ph drop": 4, "chain scission": 4, "degradation uniformity": 5, "uniform degradation": 5,
        "bulk erosion": 4, "surface erosion": 4, "water uptake": 3, "water diffusion": 3,
        "release": 2, "resorption": 3, "lifetime": 2,
        "降解": 5, "水解": 5, "侵蚀": 4, "质量损失": 4, "自催化": 5, "酸性副产物": 5,
        "均匀降解": 5, "降解均匀性": 5, "链断裂": 4, "服役时间": 2,
    },
}


def _normalize_text_for_scoring(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _score_keywords(text: str, keyword_map: dict[str, int], cap_per_keyword: int = 2) -> int:
    if not text:
        return 0
    score = 0
    for phrase, weight in keyword_map.items():
        count = text.count(phrase.lower())
        if count > 0:
            score += min(count, cap_per_keyword) * int(weight)
    return score


def _score_domain_hints(domain: dict | None) -> dict[str, int]:
    scores = {key: 0 for key in CATEGORY_DISPLAY_NAMES}
    if not isinstance(domain, dict):
        return scores

    raw_tokens = []
    for value in domain.values():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            raw_tokens.extend(str(v) for v in value)
        else:
            raw_tokens.append(str(value))
    joined = _normalize_text_for_scoring(" | ".join(raw_tokens))

    for category, kw_map in CATEGORY_KEYWORDS.items():
        scores[category] += _score_keywords(joined, kw_map, cap_per_keyword=1)

    target_property = _normalize_text_for_scoring(domain.get("target_property", ""))
    modification_type = _normalize_text_for_scoring(domain.get("modification_type", ""))
    mechanism_type = _normalize_text_for_scoring(domain.get("mechanism_type", ""))
    method_type = _normalize_text_for_scoring(domain.get("method_type", ""))

    if any(tok in target_property for tok in ["mechanical", "strength", "modulus", "stiffness"]):
        scores["mechanical_design"] += 8
    if any(tok in target_property for tok in ["tg", "tm", "thermal", "glass transition", "high tg"]):
        scores["thermal_design"] += 9
    if any(tok in target_property for tok in ["degradation", "hydrolysis", "water diffusion", "water uptake"]):
        scores["degradation_regulation"] += 9
    if any(tok in modification_type for tok in ["copolymer", "blending", "blend", "crosslink"]):
        pass
    if any(tok in mechanism_type for tok in ["erosion", "autocatalysis", "hydrolysis"]):
        scores["degradation_regulation"] += 7
    if any(tok in method_type for tok in ["dsc", "dma"]):
        scores["thermal_design"] += 3
    if any(tok in method_type for tok in ["tensile", "compression", "mechanical"]):
        scores["mechanical_design"] += 3

    if callable(coarsen_domain_to_report_hints):
        try:
            external = coarsen_domain_to_report_hints(domain)
            for cat, extra in (external.get("scores") or {}).items():
                if cat in scores:
                    scores[cat] += int(extra)
        except Exception:
            pass

    return scores


def classify_design_report_category(
    *,
    original_query: str,
    idea_prefix: str,
    domain: dict | None,
    structured_evidence_text: str,
    structured_table_text: str,
    preset_profile: dict | None = None,
) -> dict:
    """对最终 Design report 进行写作模板分类。采用规则打分，避免额外 LLM 开销。"""
    if isinstance(preset_profile, dict) and preset_profile.get("primary_category"):
        return {
            "primary_category": preset_profile.get("primary_category", "degradation_regulation"),
            "primary_display_name": preset_profile.get(
                "primary_display_name",
                CATEGORY_DISPLAY_NAMES.get(
                    preset_profile.get("primary_category", "degradation_regulation"),
                    preset_profile.get("primary_category", "degradation_regulation"),
                ),
            ),
            "secondary_categories": list(preset_profile.get("secondary_categories") or []),
            "secondary_display_names": list(preset_profile.get("secondary_display_names") or []),
            "mode": preset_profile.get("mode", "single"),
            "score_breakdown": dict(preset_profile.get("score_breakdown") or {}),
            "rationale": preset_profile.get("rationale", "Reused precomputed design profile."),
        }
    query_text = _normalize_text_for_scoring(f"{original_query}\n{idea_prefix}")
    evidence_text = _normalize_text_for_scoring((structured_evidence_text or "")[:8000])
    table_text = _normalize_text_for_scoring((structured_table_text or "")[:3000])

    scores = {key: 0 for key in CATEGORY_DISPLAY_NAMES}
    for category, kw_map in CATEGORY_KEYWORDS.items():
        scores[category] += 3 * _score_keywords(query_text, kw_map, cap_per_keyword=2)
        scores[category] += 1 * _score_keywords(evidence_text, kw_map, cap_per_keyword=2)
        scores[category] += 1 * _score_keywords(table_text, kw_map, cap_per_keyword=1)

    domain_scores = _score_domain_hints(domain)
    for category, value in domain_scores.items():
        scores[category] += int(value)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    primary_category, primary_score = ranked[0]

    if primary_score <= 0:
        primary_category = "degradation_regulation"
        primary_score = 1
        scores[primary_category] = max(scores[primary_category], 1)

    secondary_categories = []
    hybrid_cutoff = max(12, int(primary_score * 0.5))
    for category, score in ranked[1:]:
        if len(secondary_categories) >= 2:
            break
        if score >= hybrid_cutoff:
            secondary_categories.append(category)

    mode = "hybrid" if secondary_categories else "single"

    top_signals = []
    for category, score in ranked[:3]:
        if score > 0:
            top_signals.append(f"{CATEGORY_DISPLAY_NAMES[category]}={score}")
    rationale = (
        "Primary emphasis selected from weighted query/domain/evidence signals. "
        + (" | ".join(top_signals) if top_signals else "No strong lexical signal; fell back to degradation-focused default.")
    )

    return {
        "primary_category": primary_category,
        "primary_display_name": CATEGORY_DISPLAY_NAMES[primary_category],
        "secondary_categories": secondary_categories,
        "secondary_display_names": [CATEGORY_DISPLAY_NAMES[c] for c in secondary_categories],
        "mode": mode,
        "score_breakdown": scores,
        "rationale": rationale,
    }




def build_design_profile(
    *,
    original_query: str,
    idea_title: str,
    idea_mechanism: str,
    domain: dict | None,
) -> dict:
    """
    统一的 Design profile：
    - Router 阶段只分类一次
    - per-idea planner 的检索增强与最终 report prompt 复用同一 profile
    """
    base_profile = classify_design_report_category(
        original_query=original_query,
        idea_prefix=f"{idea_title}\n{idea_mechanism}",
        domain=domain,
        structured_evidence_text="",
        structured_table_text="",
    )

    primary_category = base_profile.get("primary_category", "degradation_regulation")
    secondary_categories = list(base_profile.get("secondary_categories") or [])
    active_categories = [primary_category] + secondary_categories
    active_categories = list(dict.fromkeys(active_categories))[:3]
    active_set = set(active_categories)

    extra_searches = []
    if {"thermal_design", "mechanical_design"}.issubset(active_set):
        extra_searches.append({
            "name": "thermal_mechanical_tradeoff",
            "description": "Search trade-off evidence linking Tg/crystallinity/processing window to modulus/toughness/ductility.",
            "query": (
                f"{original_query} {idea_title} "
                "biodegradable polymer Tg toughness modulus ductility crystallinity trade-off processing window"
            ),
            "top_k": 15,
        })

    if {"degradation_regulation", "mechanical_design"}.issubset(active_set):
        extra_searches.append({
            "name": "degradation_mechanical_retention",
            "description": "Search evidence on mechanical retention during hydrolytic degradation / wet-state support.",
            "query": (
                f"{original_query} {idea_title} "
                "biodegradable polymer mechanical retention during hydrolytic degradation wet-state support strength retention"
            ),
            "top_k": 15,
        })

    if {"thermal_design", "degradation_regulation"}.issubset(active_set):
        extra_searches.append({
            "name": "thermal_degradation_coupling",
            "description": "Search coupling among Tg/crystallinity/water uptake/hydrolysis/degradation.",
            "query": (
                f"{original_query} {idea_title} "
                "biodegradable polymer crystallinity Tg water uptake hydrolysis degradation coupling"
            ),
            "top_k": 15,
        })

    profile = dict(base_profile)
    profile.update({
        "active_categories": active_categories,
        "active_display_names": [CATEGORY_DISPLAY_NAMES[c] for c in active_categories],
        "extra_searches": extra_searches,
        "max_total_search_steps": 4 if len(active_categories) >= 2 else 2,
    })
    return profile

def _build_category_focus_block(report_category: dict | None) -> str:
    rc = report_category or {}
    primary = rc.get("primary_category", "degradation_regulation")
    secondary = list(rc.get("secondary_categories") or [])
    primary_name = CATEGORY_DISPLAY_NAMES.get(primary, primary)
    secondary_names = [CATEGORY_DISPLAY_NAMES.get(c, c) for c in secondary]
    secondary_line = ", ".join(secondary_names) if secondary_names else "None"

    common = f"""
DESIGN REPORT CATEGORY PROFILE:
- Primary category: {primary_name}
- Secondary categories: {secondary_line}
- Writing mode: {rc.get('mode', 'single')}

MANDATORY WRITING POLICY:
- Keep the global manuscript skeleton unchanged so downstream parsing remains stable.
- Change the EMPHASIS, comparison logic, failure analysis, and experimental plan depth according to the category profile.
- The final report should read like an experiment-guiding research manuscript, not a general overview.
- In Section 3 and Section 5, prioritize variables, controls, and decision rules that best match the primary category, while still covering any high-scoring secondary category.
"""

    blocks = {
        "mechanical_design": """
CATEGORY-SPECIFIC PRIORITIES — 力学性能设计:
- In Section 2, make the structure-property pathway explicit for stiffness, strength, toughness, ductility, and wet-state retention.
- In Section 3.5, specify dry-state vs wet-state testing logic, loading mode, specimen groups, and failure-mode comparisons.
- In Section 3.6, connect degradation monitoring to mechanical retention rather than discussing mass loss in isolation.
- In Section 4 and 5, emphasize trade-offs among stiffness, toughness, crack resistance, and mechanical support duration.
- Include practical guidance on how to tune crystallinity, reinforcement level, molecular weight, crosslink density, or porosity when they are supported by evidence.
""",
        "thermal_design": """
CATEGORY-SPECIFIC PRIORITIES — 热学性能设计:
- In Section 2, focus on Tg/Tm, chain mobility, crystallization behavior, thermal history, and processing-window logic.
- In Section 3.4, explicitly prioritize DSC, DMA, XRD, thermal stability, and morphology methods when supported.
- In Section 3.3, discuss melt-processing feasibility, annealing, solvent-removal thermal history, and thermal-process sensitivity.
- In Section 4 and 5, emphasize how thermal transitions couple to processing stability, dimensional stability, mechanical response, and degradation behavior.
- Make the report useful for deciding whether the proposed formulation can actually be fabricated reproducibly.
""",
        "degradation_regulation": """
CATEGORY-SPECIFIC PRIORITIES — 降解行为与结构调控:
- In Section 2, center the mechanism on water ingress, chain scission, autocatalysis, surface-vs-bulk erosion, and degradation heterogeneity.
- In Section 3.6, specify how to measure mass loss, molecular-weight decline, pH evolution, morphology evolution, and mechanical-retention coupling over time.
- In Section 4, separate directly supported degradation evidence from inferred lifetime expectations.
- In Section 5, explicitly discuss acidic byproduct risk, local pH drop, nonuniform degradation, hollow-core formation, and accelerated late-stage failure where relevant.
- The optimization path should clearly state how to slow, speed up, or homogenize degradation through structure, dimensions, hydrophilicity, crystallinity, end groups, or composition.
""",
    }

    cross_axis = []
    if "mechanical_design" in secondary and primary == "degradation_regulation":
        cross_axis.append("- Treat mechanical-retention-over-time as a core endpoint rather than a secondary note.")
    if "degradation_regulation" in secondary and primary == "mechanical_design":
        cross_axis.append("- Do not discuss mechanical support without explicitly stating how degradation may erode it over time.")
    if "thermal_design" in secondary:
        cross_axis.append("- Make the thermal/processability implications explicit whenever composition or degradation changes are discussed.")

    chosen = [blocks.get(primary, "")] + [blocks.get(cat, "") for cat in secondary]
    cross_block = "\nCROSS-AXIS INTEGRATION RULES:\n" + "\n".join(cross_axis) if cross_axis else ""
    return common + "\n" + "\n".join(b for b in chosen if b.strip()) + cross_block


def build_design_report_prompt(
    *,
    original_query: str,
    idea_prefix: str,
    domain: dict | None,
    structured_evidence_text: str,
    structured_table_text: str,
    reference_map_text: str,
    report_category: dict | None = None,
) -> str:
    domain_str = json.dumps(domain or {}, ensure_ascii=False, indent=2)
    category_focus_block = _build_category_focus_block(report_category)

    role_definition = (
        "You are a senior polymer materials scientist writing a manuscript-style "
        "experimental research report for a proposed degradable polymer design."
    )

    task_instruction = f"""
Task:
Generate ONE complete manuscript-style academic report in Markdown for the design idea "{idea_prefix}".

This report is the FINAL deliverable for this idea.
Write the report itself only.
Do NOT output reviewer notes, planning notes, bullet-point planning chatter, or meta-commentary.

The report must be grounded primarily in LITERATURE evidence and secondarily in STRUCTURED DATABASE / TABLE evidence if available.

STRICT EVIDENCE RULES:
1. For any specific quantitative claim, mechanism statement, processing condition, composition ratio, temperature, time, or comparison derived from literature evidence, cite the supporting source in the body using [n].
2. The final References section must list only unique Pure Clean IDs corresponding to the cited literature sources.
3. If database/table evidence is available, integrate it explicitly as database/table evidence, but do NOT invent fake literature-style source IDs for table rows.
4. If exact experimental parameters are missing, explicitly state that they are missing / TBD. Do NOT invent numeric values.
5. If evidence is indirect, weak, or incomplete, keep the language cautious and explicitly mark uncertainty.
6. Distinguish clearly between:
   - directly supported literature evidence
   - mechanistically inferred expectations
   - unresolved evidence gaps

SCORING RULES:
- You MUST provide one machine-readable score block near the end so downstream visualization continues to work.
- If evidence is weak or missing, assign conservative scores.
- High scores (>80) require strong direct evidence.
"""

    method_rule = """
METHOD WRITING RULES (STRICT):
Section 3 MUST read like an experimental "Materials and Methods" section from a real paper, not like a project memo.

FORMULATION & MATRIX DESIGN RULES (APPLIES TO ALL REPORTS):
- In Section 3.2, make the sample matrix concrete: which ratios/components vary first, which stay fixed, what the baseline/controls are, and what constitutes the first-pass DOE (Design of Experiments).
- In Section 3.3, state formulation order, mixing logic, solvent / melt route, and processing sensitivities when evidence supports them.
- Compare composition windows rather than only single formulations.

For each subsection in Section 3, do the following whenever evidence allows:
- state the experimental objective
- define the primary readouts
- identify missing parameters explicitly as TBD
- separate mechanical testing, degradation testing, and biocompatibility testing
- include a minimal statistical / comparison strategy when possible

Avoid vague wording such as "optimize appropriately" or "test several conditions". Write in a concrete manuscript style.
"""

    output_format = """
Output exactly in the following Markdown structure:

# [Professional Academic Title]

## Abstract
Provide a concise but information-dense abstract covering:
- design concept
- scientific hypothesis
- evidence level
- proposed experimental route
- key expected outcomes
- major risks
- final conclusion

## 1. Introduction
### 1.1 Scientific Background
- Briefly define the user goal and the scientific background.

### 1.2 Design Hypothesis
- State the core design hypothesis clearly.
- Explain why this idea is scientifically interesting and practically relevant.

### 1.3 Scope of This Study
- Clarify what is directly supported by evidence and what remains to be experimentally validated.

## 2. Mechanistic Rationale and Design Hypothesis
### 2.1 Mechanistic Basis
- Translate the retrieved evidence into explicit mechanistic understanding.

### 2.2 Evidence-Supported Design Rules
- State which structural or formulation variables are expected to influence:
  - mechanical properties
  - degradation behavior
  - biocompatibility
- If any axis lacks evidence, explicitly state that current evidence is insufficient.

### 2.3 Trade-Offs and Constraints
- Discuss trade-offs among stiffness, toughness, degradation rate, wet-state performance, and biological response when evidence supports such discussion.

## 3. Materials and Methods
### 3.1 Material System and Variable Definition
- Define the proposed material system and all experimentally relevant variables.
- Separate:
  - evidence-supported starting variables
  - unknown variables requiring optimization

### 3.2 Formulation / Sample Matrix Design
- Propose a conservative first-pass comparison matrix.
- State what should vary first and what should remain fixed.
- Include baseline controls.

### 3.3 Sample Preparation / Fabrication Procedure
- Write a stepwise preparation or fabrication workflow in manuscript style.
- Even if exact conditions are unavailable, describe the sequence and intended control purpose of each step.

### 3.4 Structural and Physicochemical Characterization
- Describe the minimum structure / chemistry / morphology / thermal characterization package needed.
- Explain why each method is necessary.

### 3.5 Mechanical Testing Protocol
- Define the mechanical evaluation logic.
- Specify what comparisons must be made across groups.
- Distinguish dry-state vs wet-state testing when relevant.

### 3.6 Degradation Evaluation Protocol
- Describe how degradation should be monitored.
- Include mass loss / morphology / property retention / chemical change logic when relevant.
- Distinguish short-term screening from longer-term validation.

### 3.7 Biocompatibility / Biofunction Evaluation
- Describe the minimum biological validation needed.
- If evidence is weak, explicitly state that this section is provisional.

### 3.8 Controls, Decision Criteria, and Statistical Comparison
- State what results would support, weaken, or falsify the hypothesis.
- Include a minimal comparison / repeat / statistics strategy whenever possible.
- Do NOT invent unsupported thresholds.

## 4. Results and Evidence-Based Discussion
### 4.1 Directly Supported Expectations
- Summarize the claims directly supported by highly matched literature evidence.
- If none exists, explicitly state: "No directly matched evidence found."

### 4.2 Mechanistically Inferred Expectations
- Summarize what is inferable from related but indirect evidence.

### 4.3 Evidence Gaps and Uncertainty
- Explicitly identify missing links, including where relevant:
  - lack of wet-state data
  - bulk vs scaffold mismatch
  - lack of in vivo data
  - missing degradation kinetics
  - missing long-term biocompatibility evidence

### 4.4 Comparison with Prior Systems
- Compare the proposed design with related prior material systems or methods using evidence.

## 5. Risk Analysis and Optimization Path
### 5.1 Major Failure Modes
- State the most likely failure modes.

### 5.2 Optimization Pathway
- Explain how the next experimental round should change if:
  - degradation is too fast
  - degradation is too slow
  - stiffness is too low
  - toughness is too low
  - wet-state performance is poor
  - biocompatibility is weaker than expected

## 6. Conclusion
- Give a final verdict on whether the idea is promising, risky, incremental, or highly innovative.
- State clearly whether it is recommended for further experimental validation.
- Summarize the single most important next experiment.

## 7. Machine-Readable Score Block
```json
{
  "feasibility": <0-100 integer>,
  "predictability": <0-100 integer>,
  "performance": <0-100 integer>,
  "innovation": <0-100 integer>
}
```

## 8. References
[[AUTO_REFERENCES]]
"""
    return f"""
{role_definition}

Original User Query: {original_query}
Context Focus: {idea_prefix}
Domain info:
{domain_str}

Reference Map (SOURCE IDENTITY MAP ONLY):
{reference_map_text}

Structured Literature Evidence:
{structured_evidence_text}

Structured Table / Database Evidence:
{structured_table_text}

{task_instruction}

{method_rule}

{category_focus_block}

{output_format}

⚠️ **CITATION FORMATTING RULES (STRICT)**:
The Reference Map above is provided only to identify the correct cleaned literature source IDs and deduplicate repeated sources.

1. Use [n] citations only for literature-supported claims grounded in the Structured Literature Evidence.
2. If you use structured database/table evidence, describe it explicitly as database/table evidence, but do NOT fabricate literature-style source IDs for it.
3. In the main body, cite specific mechanisms, quantitative claims, processing conditions, and comparisons from literature using [n].
4. In the References section, DO NOT generate any reference entries yourself.
5. Keep the exact placeholder [[AUTO_REFERENCES]] unchanged in the References section.
6. DO NOT output raw filenames, chunk suffixes, or file extensions.
7. If multiple evidence snippets come from the same source, keep them under the same [n].
"""


def generate_design_report_from_structured_evidence(
    *,
    original_query: str,
    idea_prefix: str,
    domain: dict | None,
    structured_evidence_text: str,
    structured_table_text: str,
    reference_map_text: str,
    ordered_source_ids: list[str],
    report_category: dict | None = None, 
    llm_callable: Callable[..., str] | None = None,
    temperature: float | None = None,
) -> str:
    """现役 design report 统一写作入口。"""
    llm = llm_callable or call_deepseek_llm_Report
    prompt = build_design_report_prompt(
        original_query=original_query,
        idea_prefix=idea_prefix,
        domain=domain,
        structured_evidence_text=structured_evidence_text,
        structured_table_text=structured_table_text,
        reference_map_text=reference_map_text,
        report_category=report_category, 
    )
    raw_response = _call_llm_with_fallback(llm, prompt, temperature=temperature)
    reasoning = strip_think_tags(raw_response)
    return reasoning


def generate_single_paper(
    user_query: str,
    idea_data: dict,
    index: int,
    llm_callable: Callable[..., str] | None = None,
    temperature: float = 0.0,
) -> str:
    result_text = str(idea_data.get("result", "") or "").strip()
    ordered_source_ids = idea_data.get("ordered_source_ids") or []

    if _looks_like_final_report(result_text):
        if ordered_source_ids:
            return normalize_citations_and_rebuild_references(result_text, ordered_source_ids)
        return result_text

    raise RuntimeError(
        "generate_single_paper() is deprecated as a generator. "
        "It may only reuse an already generated final report."
    )