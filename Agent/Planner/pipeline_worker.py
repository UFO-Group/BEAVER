# Agent/Planner/pipeline_worker.py

import os
import re
import json
import glob
import pandas as pd
from rich.console import Console

# ====================================================
# 📦 模块导入
# ====================================================
from Agent.Agent_Config.agent_config import console, STEP_LOG_DIR
from Agent.Report.docx_engine import save_text_to_docx
from Agent.Utils.file_utils import save_step_result 
from Agent.Utils.path_utils import sanitize_filename # 导入刚才分离的工具
from Agent.Report.paper_writer import remove_machine_readable_score_block, build_design_profile

try:
    from Agent.Agent_Config.deepseek_client import call_deepseek_llm_Score
except Exception:
    from Agent.Agent_Config.deepseek_client import call_deepseek_llm as call_deepseek_llm_Score


try:
    from Agent.Report.repeat_unit_figure import enrich_design_report_with_repeat_unit_figure
except Exception:
    enrich_design_report_with_repeat_unit_figure = None

# ----------------------------------------------------
# Repeat-unit figure dictionary paths
# ----------------------------------------------------
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(AGENT_ROOT, "resources")

# name_name_dict_path can be either a mapping file or a keyword txt folder.
# For your current workflow, each .txt file in this folder stores aliases/keywords,
# and the txt filename is treated as the canonical polymer name.
REPEAT_UNIT_NAME_KEYWORD_DIR = os.getenv(
    "REPEAT_UNIT_NAME_KEYWORD_DIR",
    os.path.join(RESOURCES_DIR, "Polymer_name"),
)

# CSV containing canonical polymer names and SMILES.
REPEAT_UNIT_NAME_SMILES_CSV = os.getenv(
    "REPEAT_UNIT_NAME_SMILES_CSV",
    os.path.join(RESOURCES_DIR, "Name_SMILE.csv"),
)

# Fuzzy matching is disabled by default to avoid short-abbreviation false positives
# such as CA -> Cellulose or ambiguous PCL/PC/PU mappings.
REPEAT_UNIT_ALLOW_FUZZY_NAME_MATCH = os.getenv(
    "REPEAT_UNIT_ALLOW_FUZZY_NAME_MATCH",
    "false",
).lower() in {"1", "true", "yes", "y"}

REPEAT_UNIT_FUZZY_CUTOFF = float(os.getenv(
    "REPEAT_UNIT_FUZZY_CUTOFF",
    "0.92",
))

# True: if no dictionary-derived RDKit-valid SMILES is found, skip the figure.
# This avoids misleading schematic fallback boxes.
REPEAT_UNIT_STRICT_SMILES = os.getenv(
    "REPEAT_UNIT_STRICT_SMILES",
    "true",
).lower() in {"1", "true", "yes", "y"}


# 导入 Planner 和 Executor
from Agent.Planner.planner_module import run_planner
from Agent.Planner.plan_executor import plan_executor

if not console:
    console = Console()

def _load_design_report_category(idea_sub_dir: str) -> dict:
    ctx_path = os.path.join(idea_sub_dir, "execution_context.json")
    if not os.path.exists(ctx_path):
        return {}
    try:
        with open(ctx_path, "r", encoding="utf-8") as f:
            ctx = json.load(f)
    except Exception:
        return {}

    direct = ctx.get("final_design_profile")
    if isinstance(direct, dict) and direct.get("primary_category"):
        return direct
    return {}




def _extract_json_object_safely(text: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    if not text:
        return {}

    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start:end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _clip_score_0_100(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return default


def run_chemical_validity_check(
    original_query: str,
    idea_title: str,
    idea_mechanism: str,
    result_text: str,
    report_category: dict | None = None,
    domain: dict | None = None,
    verbose: bool = False,
) -> dict:
    """
    Independent post-report Chemical Validity critic.

    Chemical Validity evaluates:
    1. whether the proposed structure/composition/mechanism is chemically and materially self-consistent;
    2. whether there is a plausible structural basis and mechanistic pathway supporting the targeted property changes;
    3. whether report-level claims are consistent with the proposed structure and supporting evidence;
    4. whether there are critical flaws contradicting design objectives, structure-property logic, or evidence base.
    """
    prompt = f"""
You are a strict polymer-chemistry and materials-science validity critic.

Your task is to evaluate the Chemical Validity of a generated candidate-level design report.
Do NOT reward fluent writing. Judge only chemistry, materials-science consistency, structure-property logic, and evidence-claim consistency.

Definition of Chemical Validity:
1. Evaluate whether the proposed structure, composition, and mechanism are chemically and materially self-consistent.
2. Check whether there is a plausible structural basis and mechanistic pathway supporting the targeted property changes.
3. Check whether report-level claims are consistent with the proposed structure and supporting evidence.
4. Identify critical flaws that contradict the design objectives, structure-property logic, degradation/performance mechanism, or evidence base.

Original user goal:
{original_query}

Candidate title:
{idea_title}

Candidate mechanism:
{idea_mechanism}

Domain:
{json.dumps(domain or {}, ensure_ascii=False, indent=2)}

Report category:
{json.dumps(report_category or {}, ensure_ascii=False, indent=2)}

Generated manuscript-style report:
\"\"\"
{str(result_text)[:14000]}
\"\"\"

Return ONLY valid JSON with this exact schema:
{{
  "chemical_validity": 0-100,
  "structure_mechanism_consistency": 0-100,
  "property_pathway_plausibility": 0-100,
  "evidence_claim_consistency": 0-100,
  "overall_chemical_validity": 0-100,
  "contradiction_detected": true or false,
  "major_red_flags": ["..."],
  "score_caps": {{
    "Feasibility": 0-100 or null,
    "Predictability": 0-100 or null,
    "Performance": 0-100 or null,
    "Innovation": 0-100 or null
  }},
  "rationale": "brief explanation"
}}

Strict scoring rules:
- If the proposed structure/composition/mechanism is chemically implausible, set chemical_validity <= 40.
- If there is no plausible structural basis or mechanistic pathway for the targeted property changes, set property_pathway_plausibility <= 40.
- If the report makes strong claims that are inconsistent with the proposed structure or evidence, set evidence_claim_consistency <= 50.
- If a major contradiction is detected, set contradiction_detected = true and cap Feasibility and Predictability at no more than 50.
- If the contradiction directly invalidates the central design objective, cap Feasibility at no more than 40.
- If no major flaw is found, score_caps may contain null values.
"""

    try:
        raw = call_deepseek_llm_Score(
            prompt,
            system_prompt="You are a strict JSON-only scientific critic. Output ONLY valid JSON.",
            temperature=0.0,
        )
    except Exception as e:
        if verbose:
            console.print(f"[dim]⚠️ Chemical Validity LLM call failed: {e}[/dim]")
        return {
            "chemical_validity": 0,
            "structure_mechanism_consistency": 0,
            "property_pathway_plausibility": 0,
            "evidence_claim_consistency": 0,
            "overall_chemical_validity": 0,
            "contradiction_detected": True,
            "major_red_flags": [f"Chemical Validity critic call failed: {e}"],
            "score_caps": {"Feasibility": 50, "Predictability": 50},
            "rationale": "Fallback result because the Chemical Validity critic failed.",
            "critic_status": "llm_call_failed",
        }

    obj = _extract_json_object_safely(raw)

    if not obj:
        return {
            "chemical_validity": 0,
            "structure_mechanism_consistency": 0,
            "property_pathway_plausibility": 0,
            "evidence_claim_consistency": 0,
            "overall_chemical_validity": 0,
            "contradiction_detected": True,
            "major_red_flags": ["Chemical Validity critic returned unparsable JSON."],
            "score_caps": {"Feasibility": 50, "Predictability": 50},
            "rationale": "Fallback result because the critic output could not be parsed.",
            "critic_status": "parse_failed",
            "raw_preview": str(raw)[:1000],
        }

    for key in [
        "chemical_validity",
        "structure_mechanism_consistency",
        "property_pathway_plausibility",
        "evidence_claim_consistency",
        "overall_chemical_validity",
    ]:
        obj[key] = _clip_score_0_100(obj.get(key, 0), default=0)

    if not isinstance(obj.get("major_red_flags"), list):
        obj["major_red_flags"] = [str(obj.get("major_red_flags", ""))] if obj.get("major_red_flags") else []

    if not isinstance(obj.get("score_caps"), dict):
        obj["score_caps"] = {}

    obj["contradiction_detected"] = bool(obj.get("contradiction_detected", False))
    obj.setdefault("critic_status", "validated")
    return obj

def process_single_idea_pipeline(args):
    """
    🔥 一个线程负责跑完一个 Idea 的主流程：
    1. 生成验证计划 (Plan)
    2. 执行检索 / 查表 / 最终报告生成 (Execute)
    3. 后处理并保存 Markdown / Word (Persist Outputs)
     
    ✅ 包含物理隔离逻辑：所有文件存入子文件夹
    ✅ 包含上下文隔离逻辑：强制禁用 Memory
    """
    # 解包参数
    idea_data, original_query, domain, memory, verbose, enable_quality_loop, root_export_path = args
     
    idx, idea = idea_data
    idea_idx = idx + 1
    idea_name_tag = f"idea{idea_idx}"
    idea_title = idea.get("idea_name", "Unknown Idea")
    idea_mechanism = idea.get("mechanism_one_liner", "")
     
    # 📁 [关键修复 1] 创建物理隔离的子文件夹 (强制清洗路径)
    safe_title_short = sanitize_filename(idea_title, max_length=30)
    idea_sub_dir = os.path.join(root_export_path, f"{idea_name_tag}_{safe_title_short}")
    os.makedirs(idea_sub_dir, exist_ok=True)

    # ---------------------------
    # 1. Plan (生成计划)
    # ---------------------------
    verification_query = (
        f"Original User Goal: {original_query}\n"
        f"Target Idea to Verify: {idea_title}\n"
        f"Core Mechanism: {idea_mechanism}\n\n"
        "Generate a verification plan following this PRIORITY:\n"
        
        # Step 1: 绝对核心 —— 查文献 (RAG FIRST) 强制要求查找理论支持、类似案例、加工条件和降解动力学
        "1. [CRITICAL] Find theoretical mechanism AND existing similar methods (Tool: search_papers).\n"

        # Step 2: 降级为可选 —— 查表 (TABLE SECOND) 只有当 Idea 涉及具体的力学/热学指标（如 >50MPa, Tg=60C）时才查表 如果是纯机理创新（如“设计新结构”），允许跳过此步
        "2. [OPTIONAL] Check Database (query_mech_table): IF and ONLY IF the idea involves standard mechanical targets  (Tensile Strength, Young's Modulus, Elongation at Berak, Glass Transition Temperature, Melting Temperature), "
        "search the database for reference values. If the idea is purely structural or novel, you can SKIP this step.\n"
        
        # Step 3: 综合评估 (MANDATORY) 强调基于“文献发现”来评估新颖性和可行性
        "3. [MANDATORY] Synthesize Evidence (reason_over_evidence): Evaluate Feasibility and Innovation based primarily on LITERATURE findings.\n"
        
        "Ignore all other ideas. Focus ONLY on this specific verification.\n"
        "STRICT RULE: Do NOT fabricate any data or tools. If information is missing, simply state it is missing."

        "ADDITIONAL STRICT REQUIREMENTS:\n"
        "- The final report must explicitly distinguish direct evidence vs indirect mechanistic support.\n"
        "- If evidence is missing, do not fill gaps with generic textbook assumptions.\n"
    )
     
    design_profile = idea.get("design_profile")
    if not isinstance(design_profile, dict) or not design_profile:
        design_profile = build_design_profile(
            original_query=original_query,
            idea_title=idea_title,
            idea_mechanism=idea_mechanism,
            domain=domain or {},
        )
    idea["design_profile"] = design_profile

    save_step_result(
        "step0_design_profile",
        "design_profile",
        design_profile,
        idea_sub_dir,
        console=console if verbose else None,
    )

    verification_query += (
        "\nFixed Design Category Profile:\n"
        f"- Primary: {design_profile.get('primary_category', 'degradation_regulation')}\n"
        f"- Secondary: {', '.join(design_profile.get('secondary_categories', [])) or 'None'}\n"
        f"- Mode: {design_profile.get('mode', 'single')}\n"
        "Use this fixed category profile consistently when deciding which evidence axes to prioritize.\n"
    )

    if verbose:
        console.print(f"[dim]Generating plan for {idea_name_tag} (Isolated)...[/dim]")
     
    # 🧠 [关键修复 2] 强制 memory=None
    idea_plan = run_planner(
        verification_query, 
        verbose=verbose,
        memory=None,      # 🚫 禁用记忆，强制隔离
        use_memory=False, 
        save_plan=True,      
        save_path=idea_sub_dir # ✅ 保存到子文件夹
    )
     
    if verbose:
        console.print(f"[dim]Executing plan for {idea_name_tag}...[/dim]")
     
    # ---------------------------
    # 2. Execute (执行检索与推理)
    # ---------------------------
    result_text = plan_executor(
        verification_query,        
        idea_plan,                  
        domain,                      
        export_csv=True,
        idea_prefix=idea_name_tag,
        memory=None,              # 🚫 禁用 Executor 的记忆读写
        allow_save_snapshot=False, 
        verbose=verbose,
        enable_quality_loop=enable_quality_loop,
        export_path=idea_sub_dir,  # ✅ 所有的 CSV/JSON 都写入子文件夹
        design_profile=design_profile,
    )

    report_category = _load_design_report_category(idea_sub_dir)
    if report_category is None:
        print(f"[DEBUG] report_category loaded as None in {idea_name_tag}, fallback to {{}}", flush=True)
        report_category = {}
    elif not isinstance(report_category, dict):
        print(f"[DEBUG] report_category loaded as {type(report_category)} in {idea_name_tag}, fallback to {{}}", flush=True)
        report_category = {}

    # ---------------------------
    # Chemical Validity Check
    # ---------------------------
    chemical_validity = run_chemical_validity_check(
        original_query=original_query,
        idea_title=idea_title,
        idea_mechanism=idea_mechanism,
        result_text=result_text,
        report_category=report_category,
        domain=domain,
        verbose=verbose,
    )

    save_step_result(
        "step4_chemical_validity_check",
        "chemical_validity_check",
        chemical_validity,
        idea_sub_dir,
        console=console if verbose else None,
    )

    # ---------------------------
    # 提取 5 维分数 (Score Extraction + Chemical Validity)
    # ---------------------------
    FINAL_SCORE_WEIGHTS = {
        "Feasibility": 0.22,
        "Predictability": 0.18,
        "Performance": 0.22,
        "Innovation": 0.13,
        "Chemical Validity": 0.25,
    }

    scores_dict = {
        "Feasibility": 0,
        "Predictability": 0,
        "Performance": 0,
        "Innovation": 0,
        "Chemical Validity": int(chemical_validity.get("overall_chemical_validity", 0)),
    }
    overall_score = 0
    
    try:
        valid_scores = []
    
        # ---------------------------
        # A. 优先解析 JSON 分数块（同时支持 fenced / unfenced）
        # ---------------------------
        parsed_from_json = False
        score_obj = None
        
        # 先找 fenced JSON
        json_candidates = re.findall(
            r"```json\s*(\{.*?\})\s*```",
            result_text,
            re.DOTALL | re.IGNORECASE
        )
        
        # 再找 unfenced JSON：优先限定在 Machine-Readable Score Block 之后
        section_match = re.search(
            r"(?is)##\s*\d+\.\s*Machine-Readable Score Block\s*(.*?)(?=##\s*\d+\.|\Z)",
            result_text,
        )
        if section_match:
            section_text = section_match.group(1)
            bare_candidates = re.findall(r"(\{.*?\})", section_text, re.DOTALL)
            json_candidates.extend(bare_candidates)
        else:
            # 兜底：全文里抓 JSON，但后面会校验 key，不会随便误吃
            bare_candidates = re.findall(r"(\{.*?\})", result_text, re.DOTALL)
            json_candidates.extend(bare_candidates)
        
        # 去重，按出现顺序保留
        seen_json = set()
        dedup_candidates = []
        for cand in json_candidates:
            c = cand.strip()
            if c and c not in seen_json:
                seen_json.add(c)
                dedup_candidates.append(c)
        
        key_map = {
            "feasibility": "Feasibility",
            "predictability": "Predictability",
            "performance": "Performance",
            "innovation": "Innovation",
        }
        
        for cand in dedup_candidates:
            try:
                obj = json.loads(cand)
                if not isinstance(obj, dict):
                    continue
        
                required = set(key_map.keys())
                if not required.issubset(set(obj.keys())):
                    continue
        
                tmp_scores = {}
                tmp_valid_scores = []
        
                for raw_key, out_key in key_map.items():
                    val = int(obj[raw_key])
                    val = max(0, min(100, val))
                    tmp_scores[out_key] = val
                    tmp_valid_scores.append(val)
        
                if len(tmp_valid_scores) == 4:
                    scores_dict.update(tmp_scores)
                    # Chemical Validity comes from the independent critic, not from the report itself.
                    scores_dict["Chemical Validity"] = int(
                        chemical_validity.get("overall_chemical_validity", 0)
                    )
                    parsed_from_json = True
                    score_obj = obj
                    break
        
            except Exception as e:
                if verbose:
                    console.print(f"[dim]⚠️ JSON score parse failed: {e}[/dim]")
    
        # ---------------------------
        # B. JSON 失败后，回退到原始文本 regex
        # ---------------------------
        if not parsed_from_json:
            patterns = {
                "Feasibility":    r"(?im)^\s*[-*]?\s*\**Feasibility\**(?:\s+Score)?\s*:\s*(?:[A-Za-z \-]+)?\(?\s*(\d{1,3})\s*/\s*100\)?",
                "Predictability": r"(?im)^\s*[-*]?\s*\**Predictability\**(?:\s+Score)?\s*:\s*(?:[A-Za-z \-]+)?\(?\s*(\d{1,3})\s*/\s*100\)?",
                "Performance":    r"(?im)^\s*[-*]?\s*\**Performance\**(?:\s+Score)?\s*:\s*(?:[A-Za-z \-]+)?\(?\s*(\d{1,3})\s*/\s*100\)?",
                "Innovation":     r"(?im)^\s*[-*]?\s*\**Innovation\**(?:\s+Score)?\s*:\s*(?:[A-Za-z \-]+)?\(?\s*(\d{1,3})\s*/\s*100\)?",
            }
    
            valid_scores = []
            for key, pat in patterns.items():
                match = re.search(pat, result_text, re.IGNORECASE)
                if match:
                    val = int(match.group(1))
                    if val <= 10:
                        val *= 10
                    val = max(0, min(100, val))
                    scores_dict[key] = val
                    valid_scores.append(val)
    
            if valid_scores:
                scores_dict["Chemical Validity"] = int(
                    chemical_validity.get("overall_chemical_validity", 0)
                )
            else:
                match_total = re.search(r"(?:Score|Confidence).*?[:=]\s*(\d+)", result_text, re.IGNORECASE)
                if match_total:
                    val = int(match_total.group(1))
                    if val <= 10:
                        val *= 10
                    val = max(0, min(100, val))
                    for k in ["Feasibility", "Predictability", "Performance", "Innovation"]:
                        scores_dict[k] = val
                    scores_dict["Chemical Validity"] = int(
                        chemical_validity.get("overall_chemical_validity", 0)
                    )
    
    except Exception as e:
        if verbose:
            console.print(f"[dim]⚠️ Score extraction error: {e}[/dim]")

    # ---------------------------
    # Apply Chemical Validity caps and recompute final overall score
    # ---------------------------
    try:
        caps = chemical_validity.get("score_caps", {}) or {}
        for dim in ["Feasibility", "Predictability", "Performance", "Innovation"]:
            cap = caps.get(dim, None)
            if cap is not None:
                scores_dict[dim] = min(
                    scores_dict.get(dim, 0),
                    _clip_score_0_100(cap, default=100),
                )

        scores_dict["Chemical Validity"] = int(
            chemical_validity.get("overall_chemical_validity", 0)
        )

        overall_score = int(round(
            sum(scores_dict[k] * FINAL_SCORE_WEIGHTS[k] for k in FINAL_SCORE_WEIGHTS)
        ))

    except Exception as e:
        if verbose:
            console.print(f"[dim]⚠️ Chemical Validity score adjustment failed: {e}[/dim]")
        scores_dict["Chemical Validity"] = int(
            chemical_validity.get("overall_chemical_validity", 0)
        )
        overall_score = int(sum(scores_dict.values()) / max(1, len(scores_dict)))
    
    # 初始数据包
    processed_data = {
        "id": idea_name_tag,
        "title": idea_title,
        "mechanism": idea_mechanism,
        "result": result_text,
        "report_summary": "Summary not available.", 
        "score": overall_score, 
        "scores_dict": scores_dict, # ✅ 包含 Innovation 和 Chemical Validity 的详细分数
        "chemical_validity": chemical_validity,
        "report_category": report_category,
        "report_primary_category": report_category.get("primary_category", ""),
        "report_secondary_categories": report_category.get("secondary_categories", []),
        "report_mode": report_category.get("mode", ""),
    }

    # ---------------------------
    # 4. Persist Final Report (result_text 已由 reason_over_evidence 直接生成)
    # ---------------------------
    if idea_sub_dir:
        paper_content = str(result_text).strip()
        paper_content = remove_machine_readable_score_block(paper_content)

        # 兜底：如果 step3 没有生成标题，补一个标题，避免 docx 太难看
        if not re.match(r'^\s*#\s+', paper_content):
            paper_content = f"# {idea_title}\n\n" + paper_content


        # -------------------------------------------------
        # Optional Design-only graphical repeat-unit figure
        # -------------------------------------------------
        # Only runs in Design idea pipelines because process_single_idea_pipeline()
        # is not used by standard Query mode. Failure here must never break report saving.
        repeat_unit_meta = None
        if enrich_design_report_with_repeat_unit_figure is not None:
            try:
                paper_content, repeat_unit_meta = enrich_design_report_with_repeat_unit_figure(
                    report_text=paper_content,
                    output_dir=idea_sub_dir,
                    original_query=original_query,
                    idea_title=idea_title,
                    idea_mechanism=idea_mechanism,
                    idea_tag=idea_name_tag,
                    name_name_dict_path=REPEAT_UNIT_NAME_KEYWORD_DIR,
                    name_smiles_csv_path=REPEAT_UNIT_NAME_SMILES_CSV,
                    placement="after_abstract",
                    allow_fuzzy_name_match=REPEAT_UNIT_ALLOW_FUZZY_NAME_MATCH,
                    fuzzy_cutoff=REPEAT_UNIT_FUZZY_CUTOFF,
                    strict_smiles=REPEAT_UNIT_STRICT_SMILES,
                    verbose=verbose,
                )
                processed_data["repeat_unit_figure"] = repeat_unit_meta
            except Exception as e:
                processed_data["repeat_unit_figure"] = {
                    "rendered": False,
                    "error": str(e),
                }
                if verbose:
                    print(f"⚠️ Repeat-unit figure generation failed: {e}", flush=True)


        # 提取摘要 (Abstract)
        summary_text = "Summary extraction failed."
        try:
            abstract_match = re.search(
                r'##\s*Abstract\s*(.*?)\s*(?=##|\Z)',
                paper_content,
                re.DOTALL | re.IGNORECASE
            )
            if abstract_match:
                summary_text = abstract_match.group(1).strip()
            else:
                summary_text = paper_content[:400].replace('#', '').strip() + "..."
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to extract summary: {e}")

        processed_data["report_summary"] = summary_text
        
        # 额外保存 score sidecar，供可视化或后处理使用
        try:
            score_json_path = os.path.join(idea_sub_dir, f"{idea_name_tag}_scores.json")
            with open(score_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "overall_score": overall_score,
                        "scores_dict": scores_dict,
                        "chemical_validity": chemical_validity,
                        "report_category": report_category,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to save score json: {e}")
                
        try:
            profile_json_path = os.path.join(idea_sub_dir, f"{idea_name_tag}_report_profile.json")
            with open(profile_json_path, "w", encoding="utf-8") as f:
                json.dump(report_category or {}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to save report profile json: {e}")

        # 保存 .md
        save_step_result(
            step_id=f"Paper_{idea_idx}",
            step_type="academic_paper",
            content=paper_content,
            step_log_dir=idea_sub_dir,
            suffix="md",
            console=None
        )

        # 保存 .docx
        safe_title = sanitize_filename(idea_title, max_length=30)
        file_name_base = f"Paper_{idea_idx}_{safe_title}"
        doc_file_path = os.path.join(idea_sub_dir, f"{file_name_base}.docx")

        save_text_to_docx(paper_content, doc_file_path)

    return processed_data
