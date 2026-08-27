import sys
import os
from typing import Any, Dict, Optional

# 获取路径：.../Agent/Intent -> .../Agent -> .../Project_Root
current_file = os.path.abspath(__file__)
intent_dir = os.path.dirname(current_file)
agent_dir = os.path.dirname(intent_dir)
project_root = os.path.dirname(agent_dir)

# 必须把根目录加入 sys.path，否则 'Agent.Agent_Config' 无法识别
if project_root not in sys.path:
    sys.path.append(project_root)
    
from Agent.Agent_Config.deepseek_client import call_deepseek_llm
from Agent.RAG.rag_core import rag_answer
from Agent.Agent_Config.agent_config import console
from .kg_manager import KGManager

# 初始化 KG (加载 JSON)
try:
    kg_inspector = KGManager(
        os.path.join(intent_dir, "loop_knowledge_base.json")
    )
    console.print("[green]✅ Knowledge Graph loaded successfully.[/green]")
except Exception as e:
    console.print(f"[red]⚠ Failed to load Knowledge Graph: {e}[/red]")
    kg_inspector = None


# ========== 1. Conductor (Quality Inspector) ==========

def _build_conductor_prompt(question: str, answer: str) -> tuple[str, str]:
    system_prompt = (
        "You are a strict scientific quality inspector specializing in degradable polymers. "
        "Output ONLY 'Yes' or 'No'."
    )
    user_prompt = f"""
[QUESTION]
{question}

[ANSWER]
{answer}

Evaluate relevance, scientific correctness, and plausibility.
If acceptable, output: Yes
Otherwise output: No
"""
    return system_prompt, user_prompt


def conductor_decide(question: str, answer: str) -> tuple[bool, str]:
    """
    Returns (Passed: bool, Reason: str)
    """
    
    # --- PHASE 1: 硬规则检查 (Knowledge Graph) --- [Conductor-KG]
    if kg_inspector:
        is_valid, reason = kg_inspector.check_fact_validity(answer)
        if not is_valid:
            console.print(f"[red][Conductor-KG] 🛑 BLOCKED:[/red] {reason}")
            return False, reason
        else:
            console.print(f"[green][Conductor-KG] passed:[/green] {reason}")

    # --- PHASE 2: 软逻辑检查 (LLM) ---
    system_prompt, user_prompt = _build_conductor_prompt(question, answer)
    resp = call_deepseek_llm(
        user_prompt,
        system_prompt=system_prompt,
    )
    decision = (resp or "").strip().lower()
    console.print(f"[blue][Conductor-LLM] raw decision:[/blue] {decision}")
    
    if decision.startswith("yes"):
        return True, "Approved by LLM inspector."
    else:
        return False, "Rejected by LLM inspector (Relevance/Logic)."


# ========== 2. Reformulator (Question Rewriter) ==========

def _build_reformulation_prompt(question: str, failure_reason: str = "") -> tuple[str, str]:
    """
    Constructs a strategic prompt for rewriting queries based on specific failure modes.
    """
    system_prompt = (
        "You are a sophisticated scientific search query optimizer. "
        "Your goal is to rewrite the input question to retrieve better, more accurate evidence "
        "from a vector database containing polymer science literature."
    )

    user_prompt = f"""
[ORIGINAL QUESTION]
{question}

[REJECTION REASON]
The previous retrieval attempt was rejected. The specific reason provided by the inspector is:
"{failure_reason}"

[OPTIMIZATION STRATEGY]
Based on the specific rejection reason above, apply ONE of the following strategies to rewrite the question:

1. **STRATEGY A: If the rejection is due to "Logic Conflict" or "KG Blocked" (e.g., A inhibits B vs A promotes B):**
   - Focus specifically on the relationship between the conflicting entities.
   - Formulate a question that explicitly asks for the interaction mechanism between these two entities to verify the truth.
   - Example: "Does High Crystallinity inhibit or promote water diffusion in PLA?"

2. **STRATEGY B: If the rejection is due to "Lack of Data" or "No numerical values found":**
   - **DO NOT** add arbitrary constraints like specific years (e.g., "2010-2023"), statistical metrics (R², SD), or specific sample sizes unless the original user explicitly asked for them.
   - **INSTEAD**, broaden the search terms. Use synonyms (e.g., "Poly(sebacic anhydride)" -> "Polyanhydrides" or "Surface-eroding polymers").
   - Ask for *qualitative* descriptions, mechanisms, or general trends if specific numbers are missing.

3. **STRATEGY C: If the rejection is due to "Irrelevance" or "Vague Answer":**
   - Add specific domain keywords related to the missing context (e.g., add "hydrolysis", "bulk erosion", "core-shell").
   - Clarify the context (e.g., "in the context of drug delivery").

[OUTPUT REQUIREMENT]
- Output ONLY the rewritten question text.
- Do not explain your strategy.
"""
    return system_prompt, user_prompt


def reformulate_question(question: str, failure_reason: str = "") -> str:
    system_prompt, user_prompt = _build_reformulation_prompt(question, failure_reason)
    new_q = call_deepseek_llm(user_prompt, system_prompt=system_prompt)
    new_q = (new_q or "").strip()
    console.print(f"[yellow][Reformulator] Old: {question} -> New: {new_q}[/yellow]")
    return new_q or question


# ========== 3. Quality-Supervised RAG Loop ==========

def rag_answer_with_quality_loop(
    query: str,
    *,
    domain: Dict | None,
    top_k: int,
    export_csv: bool,
    step_id: str,
    max_attempts: int = 3,
    # 🔥 [修改点] 新增 export_path 参数，用于接收外部传入的路径
    export_path: Optional[str] = None,
    retrieval_mode: str = "hybrid",
) -> Any:
    
    current_q = query
    last_ans: Any = None
    
    for i in range(max_attempts):
        console.print(f"[bold cyan]🌀 Loop attempt {i+1}/{max_attempts}[/bold cyan]")
        
        # 1. RAG Retrieve
        ans = rag_answer(
            current_q,
            domain=domain,
            top_k=top_k,
            export_csv=export_csv,
            step_id=f"{step_id}_try{i+1}",
            # 🔥 [修改点] 将路径透传给 rag_answer
            export_path=export_path,
            retrieval_mode=retrieval_mode,
        )
        
        # Handle dict/str format
        if isinstance(ans, dict):
            answer_text = ans.get("answer", str(ans))
        else:
            answer_text = str(ans)

        # 2. Conductor Check (KG + LLM)
        passed, reason = conductor_decide(current_q, answer_text)

        if passed:
            console.print("[green]✅ Answer Accepted.[/green]")
            return ans
        
        # 3. Reformulate
        last_ans = ans
        if i < max_attempts - 1:
            current_q = reformulate_question(current_q, failure_reason=reason)
        else:
            console.print("[red]⚠ Max attempts reached. Returning last best effort.[/red]")
            return last_ans
