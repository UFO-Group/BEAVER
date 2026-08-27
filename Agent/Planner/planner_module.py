# Agent/Planner/planner_module.py
import re
import json
import textwrap
import os
import sys
from rich.console import Console
from rich.table import Table

# ====================================================
# 📂 路径配置 & 模块导入
# ====================================================
current_file = os.path.abspath(__file__)
intent_dir = os.path.dirname(current_file)
agent_dir = os.path.dirname(intent_dir)
project_root = os.path.dirname(agent_dir)

# 必须把根目录加入 sys.path，否则 'Agent.Agent_Config' 无法识别
if project_root not in sys.path:
    sys.path.append(project_root)

# 🔥 [修改点 1] 增加 Import 容错
# 尝试导入特定的 LLM 客户端，如果不存在则使用通用的，并统一别名为 call_deepseek_llm_Planner_Module
try:
    from Agent.Agent_Config.deepseek_client import call_deepseek_llm_Planner_Module
except ImportError:
    from Agent.Agent_Config.deepseek_client import call_deepseek_llm as call_deepseek_llm_Planner_Module

from Agent.Utils.text_utils import parse_json_safely
from Agent.Utils.file_utils import save_step_result 
from Agent.Agent_Config.agent_config import STEP_LOG_DIR

console = Console()

def extract_json_block(text):
    if not text:
        return None

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate_str = match.group(1).strip() if match else text.strip()

    # 截取最外层 JSON
    if not match:
        p_obj_start = candidate_str.find('{')
        p_obj_end = candidate_str.rfind('}')
        p_arr_start = candidate_str.find('[')
        p_arr_end = candidate_str.rfind(']')

        start_idx, end_idx = -1, -1
        if p_obj_start != -1 and (p_arr_start == -1 or p_obj_start < p_arr_start):
            start_idx, end_idx = p_obj_start, p_obj_end
        elif p_arr_start != -1:
            start_idx, end_idx = p_arr_start, p_arr_end

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate_str = candidate_str[start_idx:end_idx + 1]

    if not candidate_str:
        return None

    # 先严格解析
    try:
        return json.loads(candidate_str)
    except Exception:
        pass

    # 轻度修复：去掉 // 注释、尾逗号
    repaired = re.sub(r'(?m)//.*$', '', candidate_str)
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    try:
        return json.loads(repaired)
    except Exception:
        pass

    # 最后再走你项目里的安全解析器
    try:
        return parse_json_safely(repaired)
    except Exception:
        return None

def _make_fallback_search_query(user_query: str) -> str:
    if not user_query:
        return ""

    # 设计验证流水线的格式
    m_goal = re.search(r"Original User Goal:\s*(.+)", user_query)
    m_idea = re.search(r"Target Idea to Verify:\s*(.+)", user_query)
    m_mech = re.search(r"Core Mechanism:\s*(.+)", user_query)

    if m_goal and m_idea:
        goal = m_goal.group(1).strip()
        idea = m_idea.group(1).strip()
        mech = m_mech.group(1).strip() if m_mech else ""
        return f"{goal} Verify idea: {idea}. Mechanism: {mech}".strip()

    # DESIGN_CONTEXT_JSON 兜底
    m_original = re.search(r'"original_input"\s*:\s*"([^"]+)"', user_query)
    if m_original:
        return m_original.group(1).strip()

    # 普通 question 模式
    return user_query.strip()
# ====================================================
# 1️⃣ Planner：让 LLM 拆分问题
# ====================================================
PLANNER_SYSTEM_PROMPT = """
You are a TASK PLANNER for a degradable polymer research assistant.

Your job: given the user's query, break it into 1–5 ordered atomic steps.

Available STEP TYPES (the `type` field MUST be one of the following):
- search_papers        : [PRIMARY TOOL] use semantic RAG search over the literature to collect evidence.
- query_mech_table     : [SECONDARY TOOL] query the structured mechanical-property database (CSV/SQL).
- reason_over_evidence : Synthesize evidence AND provide the FINAL ANSWER/VERIFICATION directly.

You must ALWAYS return ONLY a valid JSON object following this schema structure. The "steps" array is dynamic and should contain as many steps as needed for the query:

{
  "query_structure": "single",
  "steps": [
    {
      "id": "step1",
      "type": "search_papers",
      "description": "Retrieve relevant literature for the main question",
      "inputs": {
        "query": "...",
        "top_k": 15,
        "query_policy": "preserve_original",
        "query_rewrite_mode": "none"
      }
    },
    {
      "id": "step2",
      "type": "reason_over_evidence",
      "description": "Synthesize evidence to answer",
      "inputs": {
        "use_steps": ["step1"]
      }
    }
  ]
}

------------------------------
HIGH-PRIORITY QUESTION-MODE RULES
------------------------------
1. Preserve the user's ORIGINAL FULL QUERY whenever the query is a single well-formed scientific question.
2. Do NOT rewrite a well-formed full question into short keyword fragments unless the query is genuinely multi-part.
3. Use query splitting ONLY when the user query contains clearly separable sub-questions, sequential tasks, or different sub-goals that would benefit from independent retrieval.
4. For a single question with multiple analytical dimensions, keep the original full query as the MAIN retrieval query, and optionally add 1–3 supplementary facet-specific search steps.
5. The planner's job is STEP ORCHESTRATION, not aggressive query simplification.
6. For the MAIN retrieval step of a single question, set:
   - "query_policy": "preserve_original"
   - "query_rewrite_mode": "none"
7. For supplementary facet steps, set:
   - "query_policy": "supplementary_facet"
   - "query_rewrite_mode": "domain_append" only if that facet is short or ambiguous.
8. For truly split sub-questions, set:
   - "query_policy": "split_subquestion"
   - "query_rewrite_mode": "domain_append" if helpful.
9. Never replace a complete user question with a short keyword bag when the original question itself is already a strong retrieval query.

------------------------------
TOOL RULES
------------------------------
1. [search_papers]
   - Use the user's original full question as the primary query whenever possible.
   - Only use short keyword queries for clearly scoped lookup steps or supplementary facets.

2. [query_mech_table] INPUTS SPECIFICATION (STRICT)
   - The database tool ONLY supports specific keys. It does NOT support generic filters.
   - ONLY include this step if the user explicitly mentions specific numeric thresholds for standard properties (Tensile Strength, Young's Modulus, Elongation at Break, Glass Transition Temperature, Melting Temperature).
   - Use `min_xxx` / `max_xxx` for thresholds and `target_xxx` for approximate targets.
   - If the query is qualitative or mechanism-focused, SKIP THIS STEP.

------------------------------
TASK PATTERNS (FOLLOW STRICTLY)
------------------------------
1) PROPERTY / DESIGN queries with numeric targets
If the user's query contains BOTH target properties and explicit numeric constraints:
- step1: search_papers (topic + target properties; keep the full query if it is already precise)
- step2: query_mech_table (filter by numeric constraints)
- step3: reason_over_evidence

2) OVERVIEW / EXPLANATION queries
If the user input is a single material name or a short factual phrase:
- step1: search_papers
- step2: reason_over_evidence

3) MECHANISM / COMPARISON / SELECTION queries
For compare / explain / discuss / analyze / evaluate / choose / rank / suitability questions:
- Use ONE main search step with the original question.
- Add supplementary facet search steps ONLY if the question has multiple analytical dimensions that deserve separate evidence collection.
- Finish with reason_over_evidence.

4) DESIGN VERIFICATION / HYPOTHESIS TESTING queries
Use this pattern ONLY when the user explicitly proposes a concrete design, formulation, modification strategy, or intervention, and asks whether it is feasible or likely to work.
- step1: search_papers
- step2: query_mech_table (optional)
- step3: reason_over_evidence

[HIGH-PRIORITY DISAMBIGUATION RULE]
If the query is phrased as ranking, choosing, prioritizing, comparing, or matching among known candidate materials,
treat it as a QUESTION / COMPARISON task, NOT as DESIGN VERIFICATION,
unless the user explicitly asks for a new design, modification strategy, or optimization route.

------------------------------
QUERY STRUCTURE LABELS
------------------------------
- "single": one core scientific question or comparison.
- "single_with_facets": one core question plus multiple analytical dimensions.
- "multi_part": clearly separable sub-questions or sequential tasks.
- "unknown": use only if the structure is genuinely unclear.

------------------------------
GENERAL RULES
------------------------------
- Use ONLY the step types listed above.
- The LAST step MUST be 'reason_over_evidence'. DO NOT generate 'final_answer'.
- Steps MUST be in logical order (step1 -> step2 -> ...).
- The "description" field MUST be concise but concrete.
- The "inputs" object MUST include key parameters.
- All keys in `inputs` MUST use snake_case English.
- For dependency steps, explicitly list `use_steps`.
- Output MUST be valid JSON. No extra commentary.
""".strip()


def plan_question(
    user_query: str,
    memory=None,
    use_memory: bool = True,
    save_plan: bool = True,
    verbose: bool = True,
    query_structure: str | None = None,
    preserve_original_query: bool = False,
) -> dict:
    if verbose:
        # 如果没有 console 对象，改用 print
        try:
            console.print("\n🧠 [bold green]Calling Planner to decompose the task...[/bold green]")
        except:
            print("\n🧠 Calling Planner to decompose the task...")

    # ==========================================
    # 1. 记忆注入逻辑 (保持不变)
    # ==========================================
    prompt_context = user_query
    
    if memory and use_memory:
        try:
            situation = memory.get_situation()
            history = memory.get_memory()
            
            if "You have not yet attempted" not in situation and "No recent attempts" not in history:
                prompt_context = f"""
USER QUERY:
{user_query}

--------------------------------------------------
📜 MEMORY CONTEXT (Past Attempts & Outcomes):
{situation}

{history}
--------------------------------------------------
INSTRUCTION:
Based on the memory above, analyze why previous attempts succeeded or failed.
Avoid repeating failed parameters. Use successful strategies.
Construct the new plan now:
"""
                if verbose: print(f"⚡ Memory Injection: ON")
            else:
                if verbose: print(f"⚪ Memory is empty or fresh, skipping injection.")
                
        except Exception as e:
            print(f"⚠️ Memory retrieval failed: {e}")
            prompt_context = user_query 
    
    # ==========================================
    # 1.5. 注入 query 结构约束（仅 question 模式使用）
    # ==========================================
    if query_structure:
        guidance_lines = [
            "[QUESTION_PLANNER_CONTEXT]",
            f"query_structure={query_structure}",
            f"preserve_original_query={'true' if preserve_original_query else 'false'}",
            "RULES:",
            "- If query_structure=single, keep the original full user query as the main search query.",
            "- If query_structure=single_with_facets, keep the original full user query as the main search query and add only limited supplementary facet queries.",
            "- If query_structure=multi_part, you may split into multiple search steps, but each step should remain semantically faithful to the relevant sub-question.",
            "- Do not convert a well-formed full question into a short keyword bag unless it is a supplementary lookup query.",
            "[/QUESTION_PLANNER_CONTEXT]",
            "",
        ]
        prompt_context = "\n".join(guidance_lines) + prompt_context

    # ==========================================
    # 2. 调用 LLM
    # ==========================================
    try:
        raw_content = call_deepseek_llm_Planner_Module(
            prompt=prompt_context,
            system_prompt=PLANNER_SYSTEM_PROMPT, # 确保已导入此变量
            temperature=0.0 
        )
        
        # 打印原始输出的前200字符，方便调试
        if verbose: 
            print(f"🔍 [Planner Raw Output]: {raw_content[:200]}...")

    except Exception as e:
        if verbose: print(f"❌ LLM Call Failed: {e}")
        # 如果 API 彻底失败，设置为空字符串，触发下面的 Fallback
        raw_content = ""
    
    # ==========================================
    # 3. 解析 JSON (核心修改点)
    # ==========================================
    
    # 🔥 使用提取器，而不是直接清洗
    plan = extract_json_block(raw_content)
    
    if plan and "steps" in plan and isinstance(plan["steps"], list):
        for i, step in enumerate(plan["steps"]):
            step_type = step.get("type")
            step.setdefault("inputs", {})
    
            # ① 强制所有 search_papers 的 top_k = 15
            if step_type == "search_papers":
                step["inputs"]["top_k"] = 15
    
            # ② 自动补齐 reason_over_evidence.use_steps
            if step_type == "reason_over_evidence":
                if not step["inputs"].get("use_steps"):
                    prev_steps = [
                        s.get("id", f"step{j+1}")
                        for j, s in enumerate(plan["steps"][:i])
                        if s.get("type") in ("search_papers", "query_mech_table")
                    ]
                    if prev_steps:
                        step["inputs"]["use_steps"] = prev_steps
                        
    # 🔥 兜底机制 (Fallback Mechanism)
    # 如果提取失败，或者提取出的 JSON 没有 'steps' 字段
    if not plan or "steps" not in plan or not isinstance(plan["steps"], list):
        print(f"⚠️ [Warning] Planner failed to generate valid JSON. Using FALLBACK PLAN.")
        if verbose:
            print(f"   -> Raw Content was: {raw_content}")
        fallback_query = _make_fallback_search_query(user_query)
        
        plan = {
            "steps": [
                {
                    "id": "step1",
                    "type": "search_papers",
                    "description": "Search relevant papers based on the query",
                    "inputs": {
                        "query": fallback_query,
                        "top_k": 15,
                        "query_policy": "preserve_original",
                        "query_rewrite_mode": "none"
                    }
                },
                {
                    "id": "step2",
                    "type": "reason_over_evidence",
                    "description": "Synthesize findings to answer the question",
                    "inputs": {"use_steps": ["step1"]}
                }
            ]
        }

    # ==========================================
    # 4. 写入 Memory (保持不变)
    # ==========================================
    if memory and save_plan:
        try:
            steps_summary = " -> ".join([s.get('type', 'step') for s in plan['steps']])
            memory.store_plan({
                "reflection": "Planning task decomposition.",
                "choice": f"Plan Generated: {steps_summary}", 
                "reason": f"Query: {user_query}" 
            })
        except Exception as e:
            if verbose: print(f"⚠️ Failed to store plan: {e}")

    return plan


# ====================================================
# 2️⃣ 展示 Planner 结果
# ====================================================
def pretty_print_plan(plan: dict):
    """用 rich 把拆分结果打印得好看一点"""
    steps = plan.get("steps", [])

    table = Table(title="🧩 Task Plan – Decomposed Steps")
    table.add_column("Index", justify="center")
    table.add_column("Step ID", justify="center")
    table.add_column("Type", justify="center")
    table.add_column("Description", overflow="fold", max_width=60)
    table.add_column("Inputs", overflow="fold", max_width=60)

    for idx, step in enumerate(steps, 1):
        sid = step.get("id", f"step{idx}")
        stype = step.get("type", "unknown")
        desc = step.get("description", "").strip() or "-"
        inputs = step.get("inputs", {})

        # 格式化 inputs 为多行字符串
        inputs_str = json.dumps(inputs, ensure_ascii=False, indent=2)
        inputs_str = textwrap.indent(inputs_str, "")

        table.add_row(
            str(idx),
            sid,
            stype,
            textwrap.fill(desc, width=50),
            inputs_str
        )

    console.print()
    console.print(table)
    console.print()

    
# 🔥 [修改点 3] 外部接口增加 save_path 参数 (原 export_path)，并执行保存
def run_planner(
    user_query: str, 
    memory=None, 
    use_memory: bool = True, 
    save_plan: bool = True, 
    verbose: bool = True,
    # 🔥 修改参数名 export_path -> save_path，与其他模块一致
    save_path: str | None = None,
    query_structure: str | None = None,
    preserve_original_query: bool = False,
) -> dict:
    
    # 执行 Planning
    plan = plan_question(user_query, memory=memory, use_memory=use_memory, save_plan=save_plan, verbose=verbose, query_structure=query_structure, preserve_original_query=preserve_original_query)

    # 只有 verbose=True 才打印大表格
    if verbose:
        console.print("\n✅ [bold green]Task decomposition completed. Plan:[/bold green]")
        pretty_print_plan(plan)

    # 🔥 [修改点 4] 将生成的 Plan 保存到指定文件夹
    # 这样你就知道这一轮 Q1 到底生成了什么计划
    try:
        # 如果 save_path 有值就用它，否则用默认 STEP_LOG_DIR
        target_dir = save_path if save_path else STEP_LOG_DIR
        
        save_step_result(
            step_id="Planner", 
            step_type="task_decomposition", 
            content=plan, 
            step_log_dir=target_dir, 
            console=console if verbose else None
        )
    except Exception as e:
        if verbose: console.print(f"[dim]⚠️ Could not save plan json: {e}[/dim]")

    return plan