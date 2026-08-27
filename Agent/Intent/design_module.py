import json
import os
import sys
import re
import ast
from typing import Dict, Any, List, Optional, Tuple

current_file = os.path.abspath(__file__)
intent_dir = os.path.dirname(current_file)
agent_dir = os.path.dirname(intent_dir)
project_root = os.path.dirname(agent_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# Import 容错处理
# ==========================================
try:
    from Agent.Agent_Config.deepseek_client import (
        call_deepseek_llm,
        call_deepseek_llm_Design,
        call_deepseek_llm_Score,
    )
except ImportError:
    try:
        from Agent.Agent_Config.deepseek_client import call_deepseek_llm
        call_deepseek_llm_Design = call_deepseek_llm
        call_deepseek_llm_Score = call_deepseek_llm
    except ImportError:
        print("❌ Critical Error: Could not import deepseek_client. Please check Agent_Config.")
        call_deepseek_llm = lambda x, **k: "{}"
        call_deepseek_llm_Design = lambda x, **k: "[]"
        call_deepseek_llm_Score = lambda x, **k: "[]"


# ==========================================
# 配置常量
# ==========================================
MAX_IDEAS = 8
SCORE_WEIGHTS = {
    "feasibility": 0.35,
    "performance_potential": 0.35,
    "controllability": 0.15,
    "novelty": 0.15,
}
DEFAULT_FALLBACK_SCORE = 5.0
MAX_SCORE_RETRY = 1


# ==========================================
# JSON 清洗 / 解析工具
# ==========================================
def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_balanced_json_block(text: str, prefer_array: bool = True) -> Optional[str]:
    """
    从混杂文本里提取最外层、配平的 JSON 数组或对象。
    优先提取数组，因为 design ideas / scored ideas 本来就应返回 list。
    """
    pairs = [("[", "]"), ("{", "}")] if prefer_array else [("{", "}"), ("[", "]")]

    for open_ch, close_ch in pairs:
        start = text.find(open_ch)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return None


def _repair_common_json_issues(text: str) -> str:
    """
    轻量修复常见 LLM JSON 毛病：
    - 智能引号
    - 尾逗号
    - fence 外残留的 json 标记
    """
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'^\s*json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def clean_and_parse_json(llm_output: str):
    """
    更稳的 JSON 解析：
    1. 去 code fence / BOM / 零宽字符
    2. 优先提取最外层配平 JSON 数组
    3. 修复尾逗号等常见问题
    4. 最后再用 ast.literal_eval 兜底
    """
    if not llm_output:
        return None

    raw = str(llm_output).strip()
    candidates: List[str] = []
    candidates.append(raw)

    stripped = _strip_code_fences(raw)
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    arr = _extract_balanced_json_block(stripped, prefer_array=True)
    obj = _extract_balanced_json_block(stripped, prefer_array=False)
    for c in [arr, obj]:
        if c and c not in candidates:
            candidates.append(c)

    repaired_candidates: List[str] = []
    for c in candidates:
        repaired = _repair_common_json_issues(c)
        if repaired and repaired not in candidates and repaired not in repaired_candidates:
            repaired_candidates.append(repaired)
    candidates.extend(repaired_candidates)

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass

    for c in candidates:
        try:
            return ast.literal_eval(c)
        except Exception:
            pass

    return None


# ==========================================
# 通用辅助工具
# ==========================================
def looks_like_explicit_design_request(user_input: str) -> bool:
    q = (user_input or "").strip().lower()
    markers = [
        "design a", "design an",
        "propose a", "propose an",
        "formulate", "formulation",
        "modification strategy", "modification route",
        "surface-modification route", "blending route",
        "optimize", "optimization",
        "troubleshoot", "redesign",
        "how to improve", "how to accelerate", "how to mitigate",
        "verify the feasibility", "feasibility of this design",
        "suggest modifications",
    ]
    return any(m in q for m in markers)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_name(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _ensure_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _preview_text(text: str, max_len: int = 1000) -> str:
    text = str(text or "")
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...[truncated]"


def _build_fallback_scores(ideas: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
    fallback_results: List[Dict[str, Any]] = []
    for idea in ideas:
        original = idea if isinstance(idea, dict) else {}
        fallback_results.append({
            "idea_name": original.get("idea_name", "Unknown"),
            "score_overall": DEFAULT_FALLBACK_SCORE,
            "score_dimensions": {
                "feasibility": DEFAULT_FALLBACK_SCORE,
                "performance_potential": DEFAULT_FALLBACK_SCORE,
                "controllability": DEFAULT_FALLBACK_SCORE,
                "novelty": DEFAULT_FALLBACK_SCORE,
            },
            "pros": [],
            "cons": [reason],
            "summary": f"Fallback score due to scoring failure: {reason}",
            "scoring_status": "fallback",
            "original_idea": original,
        })
    return fallback_results


def _validate_and_rebuild_scores(
    scored_list: Any,
    ideas: List[Dict[str, Any]],
    debug: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    仅当评分结果满足以下条件时才视为有效：
    1. 是 list
    2. 与输入 ideas 一一对应
    3. 每个条目包含四个维度且分数落在 0–10
    4. Python 端重算 overall，不信任 LLM 提供的总分
    """
    if not isinstance(scored_list, list):
        return None, "parsed scoring result is not a list"

    input_map: Dict[str, Dict[str, Any]] = {}
    input_order: List[str] = []

    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        name = str(idea.get("idea_name", "")).strip()
        if not name:
            return None, "an input idea is missing idea_name"
        norm = _normalize_name(name)
        if norm in input_map:
            return None, f"duplicate input idea_name detected: {name}"
        input_map[norm] = idea
        input_order.append(norm)

    scored_map: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(scored_list):
        if not isinstance(item, dict):
            return None, f"scored item #{idx + 1} is not a dict"

        name = str(item.get("idea_name", "")).strip()
        if not name:
            return None, f"scored item #{idx + 1} missing idea_name"

        norm_name = _normalize_name(name)
        if norm_name not in input_map:
            return None, f"scored idea_name not found in input ideas: {name}"
        if norm_name in scored_map:
            return None, f"duplicate scored idea_name detected: {name}"

        dims = item.get("score_dimensions", {}) or {}
        if not isinstance(dims, dict):
            return None, f"score_dimensions is not a dict for idea: {name}"

        feasibility = _safe_float(dims.get("feasibility"))
        performance = _safe_float(dims.get("performance_potential"))
        controllability = _safe_float(dims.get("controllability"))
        novelty = _safe_float(dims.get("novelty"))

        vals = [feasibility, performance, controllability, novelty]
        if any(v is None for v in vals):
            return None, f"missing or non-numeric score dimension for idea: {name}"
        if any(v < 0 or v > 10 for v in vals):
            return None, f"score dimension out of range for idea: {name}"

        overall = round(
            SCORE_WEIGHTS["feasibility"] * feasibility
            + SCORE_WEIGHTS["performance_potential"] * performance
            + SCORE_WEIGHTS["controllability"] * controllability
            + SCORE_WEIGHTS["novelty"] * novelty,
            1,
        )

        scored_map[norm_name] = {
            "idea_name": input_map[norm_name].get("idea_name", name),
            "score_overall": overall,
            "score_dimensions": {
                "feasibility": feasibility,
                "performance_potential": performance,
                "controllability": controllability,
                "novelty": novelty,
            },
            "pros": _ensure_str_list(item.get("pros", [])),
            "cons": _ensure_str_list(item.get("cons", [])),
            "summary": str(item.get("summary", "") or "").strip(),
            "scoring_status": "validated",
            "original_idea": input_map[norm_name],
        }

    if set(scored_map.keys()) != set(input_order):
        missing = [input_map[n].get("idea_name", n) for n in input_order if n not in scored_map]
        extra = [n for n in scored_map.keys() if n not in input_order]
        return None, f"name set mismatch; missing={missing}, extra={extra}"

    rebuilt = [scored_map[norm_name] for norm_name in input_order]

    if debug:
        print(f"[Design Debug] ✅ Scoring schema validated for {len(rebuilt)} ideas.")

    return rebuilt, "ok"


# ==========================================
# 核心业务逻辑
# ==========================================
def extract_design_brief(user_input: str, debug: bool = False) -> Dict[str, Any]:
    """
    从自然语言设计问题中抽取：target_system / targets / constraints / context
    """
    prompt = (
        "You are a design-requirement extraction assistant for materials and scientific workflows.\n\n"
        "From the following DESIGN-type query, extract structured requirements.\n\n"
        "Return ONLY JSON with the following structure:\n"
        "{\n"
        '  "target_system": "short description of the target system/object",\n'
        '  "targets": [\n'
        "    {\n"
        '      "property": "property to achieve or optimize (e.g. tensile strength, Young\'s modulus, degradation time)",\n'
        '      "target_value": number or null,\n'
        '      "unit": "unit string or empty string"\n'
        "    }\n"
        "  ],\n"
        '  "constraints": [\n'
        '    "constraint 1 in English",\n'
        '    "constraint 2"\n'
        "  ],\n"
        '  "context": "1–3 sentence English summary of any given background, materials or methods mentioned in the query"\n'
        "}\n\n"
        "DESIGN query:\n"
        f"\"\"\"{user_input}\"\"\""
    )

    raw = call_deepseek_llm(prompt, system_prompt="You are a strict JSON generator.")

    if debug:
        print("\n[Design Debug] Raw design-brief LLM output available.")
        print("================= 大模型原始输出开始 =================")
        print(raw)
        print("================= 大模型原始输出结束 =================")

    res = clean_and_parse_json(raw)

    if not res or not isinstance(res, dict):
        if debug:
            print(f"[Design Debug] ❗ JSON parse failed in extract_design_brief. Raw preview: {_preview_text(raw, 300)}")
        return {
            "target_system": "System defined in query",
            "targets": [],
            "constraints": [],
            "context": user_input,
        }
    return res


def generate_design_ideas(
    user_input: str,
    design_brief: Dict[str, Any],
    n_ideas: int = MAX_IDEAS,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    发散生成多个备选设计方案
    """
    prompt = (
        "You are a creative scientific designer for biodegradable polymers, hydrogels "
        "and AI-for-science agents.\n\n"
        "ORIGINAL DESIGN QUERY:\n"
        f"\"\"\"{user_input}\"\"\"\n\n"
        "STRUCTURED DESIGN BRIEF (JSON):\n"
        f"{json.dumps(design_brief, ensure_ascii=False, indent=2)}\n\n"
        f"Propose {n_ideas} clearly different design ideas.\n"
        "Return ONLY a JSON array. Each element MUST have the following structure:\n"
        "[\n"
        "  {\n"
        '    "idea_name": "short name of the idea (English)",\n'
        '    "core_idea": "1–2 sentence summary of the main concept",\n'
        '    "components": [\n'
        "      {\n"
        '        "name": "key material / module / step",\n'
        '        "role": "its role in the system (e.g. matrix, crosslinker, reinforcement, agent module, retrieval, evaluator)"\n'
        "      }\n"
        "    ],\n"
        '    "mechanism": "3–5 sentence explanation of WHY this idea may achieve the design targets.",\n'
        '    "risks": [\n'
        '      "short sentence describing one risk or uncertainty",\n'
        '      "another risk"\n'
        "    ]\n"
        "  }\n"
        "]\n\n"
        "Do not output any explanation, only the JSON array.\n"
        "Do NOT wrap the JSON in markdown code fences.\n"
        "Do NOT prepend or append any extra text.\n"
    )

    raw = call_deepseek_llm_Design(
        prompt,
        system_prompt="You are a creative scientific designer. However, you must output the result as a strict, valid JSON array."
    )

    if debug:
        print(f"\n[Design Debug] Generated {n_ideas} ideas (Raw output received).")

    ideas = clean_and_parse_json(raw)

    if isinstance(ideas, dict):
        if "ideas" in ideas and isinstance(ideas["ideas"], list):
            ideas = ideas["ideas"]
        elif "data" in ideas and isinstance(ideas["data"], list):
            ideas = ideas["data"]

    if not isinstance(ideas, list):
        rescued = _extract_balanced_json_block(_strip_code_fences(raw), prefer_array=True)
        if rescued:
            try:
                ideas = json.loads(_repair_common_json_issues(rescued))
            except Exception:
                ideas = None

    if not ideas or not isinstance(ideas, list):
        if debug:
            cleaned_preview = _strip_code_fences(raw)[:800]
            print("[Design Debug] ❗ JSON parse failed. Creating Fallback Idea.")
            print(f"[Debug Raw Output (First 300 chars)]: {_preview_text(raw, 300)}")
            print(f"[Debug Cleaned Output (First 800 chars)]: {cleaned_preview}")
        return [{
            "idea_name": "LLM Generated Concept (Fallback)",
            "core_idea": "Directly derived from user query due to parsing error.",
            "components": [],
            "mechanism": "Please refer to the raw text output for details.",
            "risks": ["Parsing error occurred"],
            "raw_content_backup": str(raw)[:1500],
        }]

    flat_ideas: List[Dict[str, Any]] = []
    for item in ideas:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    flat_ideas.append(sub)
        elif isinstance(item, dict):
            flat_ideas.append(item)

    normalized_ideas: List[Dict[str, Any]] = []
    seen = set()

    for item in flat_ideas:
        idea_name = str(item.get("idea_name", "") or "").strip()
        core_idea = str(item.get("core_idea", "") or "").strip()
        mechanism = str(item.get("mechanism", "") or "").strip()

        if not (idea_name or core_idea or mechanism):
            continue

        dedup_key = _normalize_name(idea_name) if idea_name else core_idea[:120].lower()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        components = item.get("components", [])
        risks = item.get("risks", [])

        if not isinstance(components, list):
            components = []
        if not isinstance(risks, list):
            risks = [str(risks)]

        normalized_ideas.append({
            "idea_name": idea_name or f"Idea_{len(normalized_ideas) + 1}",
            "core_idea": core_idea,
            "components": components,
            "mechanism": mechanism,
            "risks": [str(r).strip() for r in risks if str(r).strip()],
        })

    ideas = normalized_ideas[:n_ideas]

    if debug:
        print(f"[Design Debug] Successfully parsed {len(ideas)} ideas (capped to {n_ideas}).")

    return ideas


def _build_scoring_prompt(design_brief: Dict[str, Any], ideas: List[Dict[str, Any]]) -> str:
    exact_names = [str(i.get("idea_name", "")).strip() for i in ideas if isinstance(i, dict)]
    return (
        "You are an expert evaluator for materials/system design and AI-for-science workflows.\n\n"
        "DESIGN BRIEF (JSON):\n"
        f"{json.dumps(design_brief, ensure_ascii=False, indent=2)}\n\n"
        "CANDIDATE IDEAS (JSON array):\n"
        f"{json.dumps(ideas, ensure_ascii=False, indent=2)}\n\n"
        "For EACH idea, assign 0–10 scores on:\n"
        "- feasibility\n"
        "- performance_potential\n"
        "- controllability\n"
        "- novelty\n\n"
        "Return ONLY a JSON array.\n"
        "The array length MUST be exactly equal to the number of input ideas.\n"
        "Use the EXACT same idea_name strings as the input. Do not rename, shorten, merge, or omit any idea.\n"
        "Each element MUST have this structure:\n"
        "{\n"
        '  "idea_name": "EXACT input idea_name",\n'
        '  "score_dimensions": {\n'
        '    "feasibility": 0-10 float,\n'
        '    "performance_potential": 0-10 float,\n'
        '    "controllability": 0-10 float,\n'
        '    "novelty": 0-10 float\n'
        "  },\n"
        '  "pros": ["..."],\n'
        '  "cons": ["..."],\n'
        '  "summary": "one-sentence summary"\n'
        "}\n\n"
        "IMPORTANT:\n"
        "- Evaluate strictly based on scientific principles.\n"
        "- Do not be afraid to give low scores to unrealistic ideas.\n"
        "- Do NOT include markdown fences.\n"
        "- Do NOT include explanatory text before or after JSON.\n"
        f"- Expected exact idea_name list: {json.dumps(exact_names, ensure_ascii=False)}\n"
    )


def score_design_ideas(
    design_brief: Dict[str, Any],
    ideas: List[Dict[str, Any]],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    收敛：对方案进行多维度打分 + 总评分。
    修复点：
    1. 解析失败时打印 raw 便于定位
    2. 对 scoring JSON 做 schema 校验
    3. Python 端重算 overall，避免 LLM 自带总分异常
    4. 校验失败时可自动重试一次
    5. fallback 时补全四个维度并打上 scoring_status
    """
    if ideas and isinstance(ideas, list):
        flat_ideas: List[Dict[str, Any]] = []
        for item in ideas:
            if isinstance(item, list):
                flat_ideas.extend([sub for sub in item if isinstance(sub, dict)])
            elif isinstance(item, dict):
                flat_ideas.append(item)
        ideas = flat_ideas[:MAX_IDEAS]

    if not ideas:
        if debug:
            print("[Design Debug] ⚠️ No ideas to score.")
        return []

    last_error_reason = "unknown scoring error"
    last_raw = ""

    for attempt_idx in range(MAX_SCORE_RETRY + 1):
        prompt = _build_scoring_prompt(design_brief, ideas)

        try:
            raw = call_deepseek_llm_Score(
                prompt,
                system_prompt="You are a rigorous scientific evaluator. Output ONLY valid JSON.",
                temperature=0.01,
            )
        except NameError:
            print("❌ Error: 'call_deepseek_llm_Score' not defined. Ensure imports are correct.")
            return []

        last_raw = raw

        if debug:
            print(f"\n[Design Debug] Scoring completed. Attempt {attempt_idx + 1}/{MAX_SCORE_RETRY + 1}")
            print("[Design Debug] Raw scoring output preview:")
            print(_preview_text(raw, 1200))

        try:
            scored_list = clean_and_parse_json(raw)
        except NameError:
            print("❌ Error: 'clean_and_parse_json' not defined. Using raw json loads.")
            try:
                scored_list = json.loads(raw)
            except Exception:
                scored_list = []

        validated, reason = _validate_and_rebuild_scores(scored_list, ideas, debug=debug)
        if validated is not None:
            final_results = validated
            final_results.sort(key=lambda x: x.get("score_overall", 0), reverse=True)
            return final_results

        last_error_reason = reason
        if debug:
            print(f"[Design Debug] ❗ Scoring validation failed on attempt {attempt_idx + 1}: {reason}")

    if debug:
        print("[Design Debug] ❗ All scoring attempts failed. Using fallback scores.")
        print(f"[Design Debug] Final failure reason: {last_error_reason}")
        print("[Design Debug] Final raw scoring output preview:")
        print(_preview_text(last_raw, 1500))

    fallback_results = _build_fallback_scores(ideas, last_error_reason)
    fallback_results.sort(key=lambda x: x.get("score_overall", 0), reverse=True)
    return fallback_results


def build_planner_view_top_ideas(
    top_ideas: List[Dict[str, Any]],
    max_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    提取给 Planner 使用的精简信息视图。
    """
    planner_view: List[Dict[str, Any]] = []

    for idx, idea in enumerate(top_ideas[:max_n]):
        if not isinstance(idea, dict):
            continue

        idea_name = idea.get("idea_name", "") or ""
        score_overall = idea.get("score_overall", 0)
        score_dims = idea.get("score_dimensions", {}) or {}

        original = idea.get("original_idea", {}) or {}
        summary = idea.get("summary", "") or ""

        core_goal = original.get("core_idea", "") or summary

        mech_text = original.get("mechanism", "") or ""
        mechanism_one_liner = ""
        if isinstance(mech_text, str) and mech_text.strip():
            tmp = mech_text.strip()
            tmp = tmp.split("。")[0]
            tmp = tmp.split(".")[0]
            mechanism_one_liner = tmp.strip()

        risks = original.get("risks", []) or []
        cons = idea.get("cons", []) or []

        risk_one_liner = ""
        if isinstance(risks, list) and risks:
            risk_one_liner = str(risks[0])
        elif isinstance(cons, list) and cons:
            risk_one_liner = str(cons[0])

        pros = idea.get("pros", []) or []
        if not isinstance(pros, list):
            pros = [str(pros)]
        if not isinstance(cons, list):
            cons = [str(cons)]

        pros_short = [str(p) for p in pros[:2]]
        cons_short = [str(c) for c in cons[:2]]

        planner_view.append({
            "rank_index": idx + 1,
            "idea_name": idea_name,
            "score_overall": score_overall,
            "score_dimensions": {
                "feasibility": score_dims.get("feasibility", None),
                "performance_potential": score_dims.get("performance_potential", None),
            },
            "core_goal": core_goal,
            "mechanism_one_liner": mechanism_one_liner,
            "risk_one_liner": risk_one_liner,
            "pros_short": pros_short,
            "cons_short": cons_short,
            "scoring_status": idea.get("scoring_status", "unknown"),
        })

    return planner_view


def design_thinking_pipeline(user_input: str, debug: bool = False):
    """
    设计类问题：发散 + 收敛 的完整流程
    返回: (design_brief, ranked_ideas)
    """
    if not looks_like_explicit_design_request(user_input):
        if debug:
            print("[Design Debug] Skip design pipeline: query does not look like an explicit design request.")
        return {
            "target_system": "",
            "targets": [],
            "constraints": [],
            "context": user_input,
            "warning": "Skip design pipeline: non-explicit design request",
        }, []

    design_brief = extract_design_brief(user_input, debug=debug)
    ideas = generate_design_ideas(user_input, design_brief, n_ideas=MAX_IDEAS, debug=debug)

    if debug:
        print("\n[Design Debug] ===== DIVERGENT IDEAS (raw) =====")

    ranked_ideas = score_design_ideas(design_brief, ideas, debug=debug)

    if debug:
        print("\n[Design Debug] ===== CONVERGENT SCORED IDEAS (ranked) =====")

    return design_brief, ranked_ideas
