# Agent/Intent/router_module.py

import re
import sys
import os
import json
from typing import Literal, TypedDict, Dict, List, Any, Optional, Union
from rich.console import Console

current_file = os.path.abspath(__file__)
intent_dir = os.path.dirname(current_file)      
agent_dir = os.path.dirname(intent_dir)        
project_root = os.path.dirname(agent_dir)       
# 将根目录加入系统路径
if project_root not in sys.path:
    sys.path.append(project_root)

from Agent.Agent_Config.deepseek_client import call_deepseek_llm_Intent
from Agent.Planner.planner_module import run_planner
from Agent.Utils.file_utils import save_step_result 
from Agent.Intent.design_module import design_thinking_pipeline, build_planner_view_top_ideas
from Agent.Intent.domain_router import classify_domain
try:
    from Agent.Report.paper_writer import build_design_profile
except ImportError:
    from paper_writer import build_design_profile

# 🔥 [新增] 引入强力解析工具 (请确保 Agent/Utils/text_utils.py 存在该函数)
try:
    from Agent.Utils.text_utils import parse_json_safely
except ImportError:
    # 简单的 fallback，防止导入失败报错
    def parse_json_safely(text):
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'```(?:json)?', '', text).strip()
        return json.loads(text)

console = Console()

# ===== 标签定义 =====

# 老的 IntentType 先保留（方便兼容）
IntentType = Literal["Question", "Design", "Unknown"]

class TurnDecomposeResult(TypedDict):
    is_chat: bool
    is_question: bool
    is_design: bool
    reason: str

# ===== 新的 Decomposer：输出三个标签 =====

def _looks_like_explicit_design(q: str) -> bool:
    q = (q or "").strip().lower()
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


def _looks_like_selection_or_query(q: str) -> bool:
    q = (q or "").strip().lower()

    starters = (
        "what ", "why ", "how ", "which ",
        "compare ", "explain ", "discuss ",
        "evaluate ", "analyze ",
        "for a ", "for an ", "if the target application is "
    )

    markers = [
        "how should one choose among",
        "choose among",
        "how would you rank",
        "rank ",
        "prioritize first",
        "more suitable",
        "build a decision framework",
        "compare at least three candidate systems",
    ]

    return q.startswith(starters) or any(m in q for m in markers)


def _force_question_override(q: str) -> bool:
    q = (q or "").strip().lower()
    force_patterns = [
        "build a decision framework",
        "which biodegradable polymer family would you prioritize first",
        "how should one choose among",
        "which biodegradable polymers are more suitable",
        "how would you rank pla, pbs, pbat-containing blends, and starch-containing blends",
    ]
    return any(p in q for p in force_patterns)


def classify_query_structure(query: str) -> Dict[str, str]:
    """
    Rule-based query structure classifier.
    Labels:
      - single
      - single_with_facets
      - multi_part
    Avoids adding another LLM call before planning.
    """
    q = (query or "").strip()
    ql = q.lower()

    sequential_markers = [
        " and then ", " then ", " first ", " second ", " third ",
        " finally ", " followed by ", " before ", " after ",
        "step 1", "step 2", "identify ", "propose ", "estimate ", "assess ",
    ]
    multi_task_patterns = [
        "compare ", " explain how ", " evaluate whether ", " discuss whether ",
        "analyze whether ", " propose how ", "design and", "compare and explain",
        "summarize and explain", "identify .* and .* explain",
    ]
    facet_markers = [
        "considering ", "in terms of ", "with respect to ", "based on ",
        "build the answer around", "focus on ", "from the perspective of ",
        "around ", "covering ", "including ", "with emphasis on ",
    ]

    has_sequential = any(m in ql for m in sequential_markers)
    has_multi_task = any(m in ql for m in [
        "compare and explain", "evaluate and propose", "identify and explain",
        "summarize and compare", "compare, then", "and then explain", "then discuss",
    ])
    if has_sequential or has_multi_task:
        return {"label": "multi_part", "reason": "Detected sequential or separable sub-tasks."}

    comma_segments = [seg.strip() for seg in re.split(r",|;", q) if seg.strip()]
    has_facets = any(m in ql for m in facet_markers)
    if has_facets and len(comma_segments) >= 2:
        return {"label": "single_with_facets", "reason": "One core question with multiple analytical dimensions."}

    if ("which " in ql or "how " in ql or "why " in ql or "what " in ql or "does " in ql or "is " in ql) and not has_sequential:
        return {"label": "single", "reason": "Single scientific ask with no clear sequential sub-tasks."}

    return {"label": "single", "reason": "Defaulted to a single-question structure."}
    
def decompose_turn(user_input: str) -> TurnDecomposeResult:
    """
    Decomposer：
    给当前 user_input 打三个标签：
    - is_chat        : 闲聊 / 问候 / 情绪表达
    - is_question : factual / mechanistic / procedural 问题
    - is_design    : design / creative / optimization 问题
    """
    prompt = f"""
You are a scientific turn classifier (Decomposer agent).

Your job is to classify the USER INPUT into EXACTLY ONE of three categories:

1. is_chat:
   - casual conversation, greetings, small talk.

2. is_question:
   - factual / mechanistic / comparative / review / selection / ranking scientific queries.
   - requires retrieving and synthesizing existing knowledge.
   - includes:
     * compare / explain / discuss / analyze / evaluate
     * material selection among known candidates
     * ranking / prioritization among known materials
     * suitability matching for a given application
     * decision frameworks and boundary-condition analysis

3. is_design:
   - ONLY when the user explicitly asks to propose, design, formulate, optimize,
     modify, troubleshoot, or recommend a NEW strategy / intervention / formulation.
   - includes:
     * "design a ..."
     * "propose a strategy ..."
     * "suggest modifications ..."
     * "how to improve / accelerate / mitigate ..."
     * "verify the feasibility of a concrete design"

### EXAMPLES:
User: "How would you rank PLA, PBS, and PBAT for food packaging?"
Output: {{"is_chat": false, "is_question": true, "is_design": false, "reason": "Ranking predefined existing materials is a standard query, not inventing a new formulation."}}

User: "Which biodegradable polymer family would you prioritize for a soft-tissue scaffold?"
Output: {{"is_chat": false, "is_question": true, "is_design": false, "reason": "Selecting/prioritizing an existing material category is a standard knowledge query."}}

User: "Design a novel biodegradable polymer strategy for a wound-healing scaffold."
Output: {{"is_chat": false, "is_question": false, "is_design": true, "reason": "Explicitly asking to invent/design a new formulation strategy."}}

IMPORTANT NEGATIVE RULES:
- The following are usually is_question, NOT is_design:
  * "Which ... is more suitable ..."
  * "How should one choose among ..."
  * "How would you rank ..."
  * "Which family would you prioritize first ..."
  * "Build a decision framework ..."
- Choosing, ranking, prioritizing, or comparing known candidates is NOT design by itself.
- is_design requires an explicit request for intervention, modification, optimization,
  or creation of a candidate strategy/system.
  
USER INPUT:
\"\"\"{user_input}\"\"\"

IMPORTANT:
- Exactly ONE of is_chat, is_question, is_design MUST be true.

Strictly output JSON in this format:
{{
  "is_chat": true or false,
  "is_question": true or false,
  "is_design": true or false,
  "reason": "Brief explanation."
}}
"""
    raw = call_deepseek_llm_Intent(prompt, sys_prompt = "You are a strict JSON generator. Output ONLY JSON.")

    if not raw:
        data = {}
    else:
        data = parse_json_safely(raw)
        
    # ---- 防御式兜底 ----
    def _to_bool(x: Any) -> bool:
        if isinstance(x, bool): return x
        if isinstance(x, str): return x.strip().lower() == "true"
        return False

    is_chat = _to_bool(data.get("is_chat", False))
    is_question = _to_bool(data.get("is_question", False))
    is_design = _to_bool(data.get("is_design", False))
    reason = str(data.get("reason", "")).strip()

    # ---- 兜底：保证三者中有且只有一个 True ----
    flags = [is_chat, is_question, is_design]
    true_count = sum(1 for f in flags if f)
    lower_q = user_input.lower()
    
    has_explicit_design = _looks_like_explicit_design(lower_q)
    has_selection_or_query = _looks_like_selection_or_query(lower_q)
    force_question = _force_question_override(lower_q)
    
    if true_count == 0:
        if force_question:
            is_chat = False
            is_question = True
            is_design = False
        elif has_explicit_design:
            is_chat = False
            is_question = False
            is_design = True
        elif has_selection_or_query or any(k in lower_q for k in ["what", "why", "how", "which", "compare", "explain", "discuss", "?"]):
            is_chat = False
            is_question = True
            is_design = False
        else:
            is_chat = True
            is_question = False
            is_design = False
    
    elif true_count > 1:
        # question 与 design 冲突时：默认 question 优先
        # 只有明确 design 请求才保留 design
        if is_question and is_design:
            if has_explicit_design and not force_question:
                is_chat = False
                is_question = False
                is_design = True
            else:
                is_chat = False
                is_question = True
                is_design = False
        elif is_question:
            is_chat = False
            is_design = False
        elif is_design:
            is_chat = False
            is_question = False
        else:
            is_chat = True
            is_question = False
            is_design = False
    
    # ===== hard override =====
    if is_design and not has_explicit_design:
        if force_question or has_selection_or_query:
            is_chat = False
            is_question = True
            is_design = False
            if not reason:
                reason = "Override: selection/ranking/comparative scientific query."

    return TurnDecomposeResult(
        is_chat=is_chat,
        is_question=is_question,
        is_design=is_design,
        reason=reason,
    )

# ===== 给 Planner 的 DESIGN 上下文格式化 =====

def format_for_planner_with_design(
    original_input: str,
    design_brief: Dict[str, Any],
    top_ideas: List[Dict[str, Any]],
) -> str:
    # 用精简视图给 Planner
    planner_view_ideas = build_planner_view_top_ideas(top_ideas, max_n=3)

    payload = {
        "original_input": original_input,
        "design_brief": design_brief,
        "top_ideas_for_planner": planner_view_ideas,
    }

    return (
        "You are a TASK PLANNER. The following JSON contains:\n"
        "- the user's original DESIGN question,\n"
        "- a structured design brief (targets + constraints),\n"
        "- the TOP 3 design ideas.\n\n"
        "Your job is to create an evidence-based step plan using ONLY allowed step types.\n"
        "Return ONLY the JSON step plan.\n"
        "[DESIGN_CONTEXT_JSON]\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


# ===== 主入口：Router =====

def answer_with_router(user_input: str, verbose: bool = True, save_path: str = None) -> Dict[str, Any]:
    # 1. 确定保存路径 (确保是传入的 save_path)
    current_log_dir = save_path if save_path else "step_logs_fallback"
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # 2. Decomposer 打标签
    tags = decompose_turn(user_input)
    is_chat = tags["is_chat"]
    is_question = tags["is_question"]
    is_design = tags["is_design"]

    if verbose:
        console.print("\n[Router Debug] ===== TURN TAGS =====")
        console.print(json.dumps(tags, ensure_ascii=False, indent=2))

    # ======== ① Chat ========
    if is_chat:
        result = {
            "route_type": "chat",
            "tags": tags,
            "original_input": user_input,
            "plan": None, 
            "message": "Chat mode.",
        }
        # 保存结果
        if save_path:
            save_step_result("step0", "router_result", result, current_log_dir)
        return result

    # ======== ② 非 chat ========
    domain = classify_domain(user_input)
    result: Dict[str, Any]

    # === 设计类 ===
    if is_design:
        # 1. Router 只负责：brief + 发散 ideas + top3 粗筛
        #    不再生成第二层 planner，避免与后续 per-idea planner 语义冲突。
        design_brief, ranked_ideas = design_thinking_pipeline(user_input, debug=verbose)
        all_generated_ideas = ranked_ideas if ranked_ideas else []
        top_ideas = ranked_ideas[:3] if ranked_ideas else []

        for idea in all_generated_ideas:
            if not isinstance(idea, dict):
                continue
            original_idea = idea.get("original_idea", {}) or {}
            mechanism_text = original_idea.get("mechanism", "") or idea.get("summary", "") or ""
            design_profile = build_design_profile(
                original_query=user_input,
                idea_title=idea.get("idea_name", ""),
                idea_mechanism=mechanism_text,
                domain=domain or {},
            )
            idea["design_profile"] = design_profile

        planner_view_top_ideas = build_planner_view_top_ideas(top_ideas)

        if save_path:
            save_step_result("step0_design_brief", "design_brief", design_brief, current_log_dir, console=console)
            save_step_result("step0_design_ideas", "design_ideas", all_generated_ideas, current_log_dir, console=console)
            save_step_result("step0_design_top_ideas", "design_top_ideas", top_ideas, current_log_dir, console=console)
            save_step_result("step0_design_planner_view", "design_planner_view", planner_view_top_ideas, current_log_dir, console=console)

        if verbose:
            console.print("\n[Router Debug] ===== DESIGN BRIEF & IDEAS Generated =====")
            console.print("[cyan]Router now stops at idea generation / ranking. Per-idea planner runs downstream only.[/cyan]")

        result = {
            "route_type": "design",
            "tags": tags,
            "domain": domain,
            "original_input": user_input,
            "design_brief": design_brief,
            "top_ideas": top_ideas,
            "all_ideas": all_generated_ideas,
            "planner_view_top_ideas": planner_view_top_ideas,
            "plan": None,
        }

    # === 问题类 ===
    elif is_question:
        query_structure = classify_query_structure(user_input)
        if verbose:
            console.print("\n[Router Debug] ===== QUERY STRUCTURE =====")
            console.print(json.dumps(query_structure, ensure_ascii=False, indent=2))
        try:
            plan = run_planner(
                user_input,
                verbose=verbose,
                save_path=current_log_dir,
                query_structure=query_structure.get("label"),
                preserve_original_query=True,
            )
        except TypeError:
            plan = run_planner(user_input, verbose=verbose)
            
        result = {
            "route_type": "question",
            "tags": tags,
            "domain": domain,
            "original_input": user_input,
            "query_structure": query_structure,
            "plan": plan,
        }

    # === 兜底 ===
    else:
        query_structure = classify_query_structure(user_input)
        try:
            plan = run_planner(
                user_input,
                verbose=verbose,
                save_path=current_log_dir,
                query_structure=query_structure.get("label"),
                preserve_original_query=True,
            )
        except TypeError:
            plan = run_planner(user_input, verbose=verbose)

        result = {
            "route_type": "fallback",
            "tags": tags,
            "domain": domain,
            "original_input": user_input,
            "query_structure": query_structure,
            "plan": plan,
        }

    # ✅ 统一保存 Router 最终结果
    if save_path:
        save_step_result(
            step_id="step0_router_final", # 区分文件名，避免覆盖
            step_type="router_result",
            content=result,
            step_log_dir=current_log_dir,
            console=console
        )

    return result

if __name__ == "__main__": 
    q = input("Question:\n> ").strip() 
    # 测试路径
    test_path = os.path.join(agent_dir, "Session_Runs", "TEST_ROUTER")
    result = answer_with_router(q, verbose=True, save_path=test_path) 
    print("\nDone.")
