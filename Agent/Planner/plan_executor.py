import textwrap
import json
import os
import re
import sys
from rich.console import Console

# ====================================================
# 📂 路径配置
# ====================================================
current_file = os.path.abspath(__file__)
intent_dir = os.path.dirname(current_file)
agent_dir = os.path.dirname(intent_dir)
project_root = os.path.dirname(agent_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# ====================================================
# 📦 模块导入
# ====================================================
from Agent.Agent_Config.agent_config import console, STEP_LOG_DIR
from Agent.Agent_Config.deepseek_client import call_deepseek_llm
from Agent.Utils.file_utils import save_step_result
# 🔥 [修改点 1] 引入 2D 和 3D 画图函数，以及 PCA
from Agent.Utils.table_plotter import plot_ashby_chart, plot_ashby_3d_chart, plot_pca_chart
from Agent.RAG.rag_core import rag_answer, retrieve_and_rerank_evidence
try:
    from Agent.Report.paper_writer import (
        generate_design_report_from_structured_evidence,
        classify_design_report_category,
    )
except ImportError:
    from paper_writer import (
        generate_design_report_from_structured_evidence,
        classify_design_report_category,
    )

# 尝试导入高级模块 (带容错)
try:
    from Agent.Quality_loop.quality_loop import rag_answer_with_quality_loop
except ImportError:
    rag_answer_with_quality_loop = None

try:
    # 🔥 [已修正] 正确导入 MECH_DB_PATH，防止 NameError
    from .table_query import run_mech_table_query, MECH_DB_PATH
except ImportError:
    run_mech_table_query = None
    MECH_DB_PATH = None

if not console:
    console = Console()
def _clean_source_id(raw: str) -> str:
    raw = str(raw or "").strip()
    raw = re.sub(r"_段\d+(?:-\d+)?", "", raw)
    raw = re.sub(r"\.txt$", "", raw)
    raw = re.sub(r"_\d+\.npy$", "", raw)
    return raw or "Unknown_Source"


def _get_source_id(item: dict) -> str:
    if not isinstance(item, dict):
        return "Unknown_Source"
    return (
        str(item.get("source_id_clean", "")).strip()
        or str(item.get("chunk_file_id_raw", "")).strip()
        or _clean_source_id(item.get("filename", item.get("source", "Unknown_Source")))
    )


def _get_evidence_text(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    # 按常见字段顺序兜底，增加 page_content 和 snippet 防止漏取
    text = (
        item.get("evidence")
        or item.get("text")
        or item.get("content")
        or item.get("page_content") 
        or item.get("snippet")     
        or item.get("chunk_text")
        or item.get("text_for_emb")
        or ""
    )
    return str(text).strip()


def _build_structured_evidence_text(
    evd: dict,
    max_sources: int | None = None,
    max_snippets_per_source: int | None = None,
    max_chars: int | None = None,
):
    """
    把上游 evd 整理成稳定编号的 source-grouped evidence。
    返回:
      evidence_text: 给 LLM 的证据文本
      ref_map_text:  给 LLM 的参考文献编号映射
      ordered_source_ids: 按编号顺序排列的 source_id 列表
    """
    source_groups = {}
    source_order = []

    for step_name, step_val in evd.items():
        if not isinstance(step_val, dict):
            continue

        items = step_val.get("evidence_items", [])
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            source_id = _get_source_id(item)
            snippet = _get_evidence_text(item)
            if not snippet:
                continue

            snippet = re.sub(r"\s+", " ", snippet).strip()
            snippet = snippet[:max_chars]

            if source_id not in source_groups:
                source_groups[source_id] = {
                    "steps": [],
                    "snippets": [],
                }
                source_order.append(source_id)

            if step_name not in source_groups[source_id]["steps"]:
                source_groups[source_id]["steps"].append(step_name)

            # 去重，避免同一段重复塞给模型
            if snippet not in source_groups[source_id]["snippets"]:
                if (
                    max_snippets_per_source is None
                    or len(source_groups[source_id]["snippets"]) < max_snippets_per_source
                ):
                    source_groups[source_id]["snippets"].append(snippet)

    # 限制最大 source 数量，避免 prompt 爆炸
    source_order = source_order[:max_sources]

    ref_lines = []
    evidence_blocks = []

    for idx, source_id in enumerate(source_order, start=1):
        group = source_groups[source_id]
        ref_lines.append(f"[{idx}] {source_id}")

        snippets_text = []
        for j, s in enumerate(group["snippets"], start=1):
            snippets_text.append(f"Snippet {j}: {s}")

        block = (
            f"Evidence [{idx}]:\n"
            f"Source ID: {source_id}\n"
            f"From step(s): {', '.join(group['steps'])}\n"
            f"{chr(10).join(snippets_text)}"
        )
        evidence_blocks.append(block)

    ref_map_text = "\n".join(ref_lines) if ref_lines else "[1] Unknown_Source"
    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No structured evidence available."
    return evidence_text, ref_map_text, source_order

def _build_structured_table_text(evd: dict, max_rows_per_step: int = 8, max_chars: int = 1200):
    """
    把 query_mech_table 返回的 list[dict] 整理成可直接给 LLM 的表格证据文本。
    注意：表格证据不进入 References 编号映射，只作为补充证据提供给模型。
    """
    table_blocks = []

    for step_name, step_val in evd.items():
        if not isinstance(step_val, list):
            continue

        rows = step_val[:max_rows_per_step]
        if not rows:
            continue

        for i, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue

            kv_pairs = []
            for k, v in row.items():
                if v is None:
                    continue
                v_str = str(v).strip()
                if not v_str:
                    continue
                kv_pairs.append(f"{k}: {v_str}")

            if not kv_pairs:
                continue

            row_text = "; ".join(kv_pairs)
            row_text = row_text[:max_chars]

            table_blocks.append(
                f"Table Evidence ({step_name}, row {i}): {row_text}"
            )

    return "\n".join(table_blocks) if table_blocks else "No structured table evidence available."
    
def _normalize_citations_and_rebuild_references(
    text: str,
    ordered_source_ids: list[str],
    default_heading: str = "5. **References**"
) -> str:
    if not text or not isinstance(text, str):
        return text

    if not ordered_source_ids:
        return text

    old_ref_map = {idx: sid for idx, sid in enumerate(ordered_source_ids, start=1) if sid}
    if not old_ref_map:
        return text

    cite_pat = re.compile(r'\[\s*([\d\s,\-&]+)\s*\]')

    def parse_nums(match_str):
        nums = set()
        clean_str = match_str.replace('&', ',')
        for part in clean_str.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                try:
                    start, end = part.split('-', 1)
                    nums.update(range(int(start), int(end) + 1))
                except ValueError:
                    pass
            else:
                try:
                    nums.add(int(part))
                except ValueError:
                    pass
        return sorted(nums)

    # 只从“正文”里提取引用顺序，因此先去掉占位符之后的区域
    text_for_scan = text
    if "[[AUTO_REFERENCES]]" in text_for_scan:
        text_for_scan = text_for_scan.split("[[AUTO_REFERENCES]]", 1)[0]

    first_appearance = []
    seen = set()

    for m in cite_pat.finditer(text_for_scan):
        nums = parse_nums(m.group(1))
        for n in nums:
            if n in old_ref_map and n not in seen:
                first_appearance.append(n)
                seen.add(n)

    # 没抓到正文 citation 时，不做 destructive 清理
    if not first_appearance:
        if "[[AUTO_REFERENCES]]" in text:
            return text.replace(
                "[[AUTO_REFERENCES]]",
                "- No cited literature sources identified."
            )
        return text

    old_to_new = {old: new for new, old in enumerate(first_appearance, start=1)}

    def _replace_citation_block(match: re.Match) -> str:
        nums = parse_nums(match.group(1))
        mapped = []
        for n in nums:
            if n in old_to_new:
                mapped.append(old_to_new[n])
        if not mapped:
            return match.group(0)
        mapped = sorted(set(mapped))
        return "[" + ", ".join(str(x) for x in mapped) + "]"

    # 先全局规范正文里的 citation 编号
    new_text = cite_pat.sub(_replace_citation_block, text)

    # 构建你自己的最终 References
    ref_items = []
    for old_id in first_appearance:
        new_id = old_to_new[old_id]
        source_id = old_ref_map[old_id]
        ref_items.append(f"- [{new_id}] {source_id}")
    
    rebuilt_ref_items = "\n".join(ref_items)
    
    if "[[AUTO_REFERENCES]]" in new_text:
        return new_text.replace("[[AUTO_REFERENCES]]", rebuilt_ref_items)
    
    rebuilt_refs = default_heading + "\n" + rebuilt_ref_items

    # 方案B：兜底，仅删除“文末”的旧 References 块，再追加新的
    tail_ref_pat = re.compile(
        r'(?is)\n{0,3}(?:#+\s*)?(?:\*{0,2}\s*)?(?:\d+\.\s*)?(?:\*{0,2}\s*)References(?:\*{0,2})?\s*:?\s*\n(?:[-*]\s*\[\d+\].*\n?)+\s*$'
    )

    cleaned_text = new_text
    if tail_ref_pat.search(cleaned_text):
        cleaned_text = tail_ref_pat.sub("", cleaned_text).rstrip()

    return cleaned_text.rstrip() + "\n\n" + rebuilt_refs

def _clean_query_formatting(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    # HTML sub/sup -> plain text
    text = re.sub(r'<\s*sub\s*>(.*?)<\s*/\s*sub\s*>', r'_\1', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*sup\s*>(.*?)<\s*/\s*sup\s*>', r'^\1', text, flags=re.IGNORECASE)

    # strip common inline html tags
    text = re.sub(r'</?(i|b|br|em|strong)\s*/?>', '', text, flags=re.IGNORECASE)

    # normalize latex delimiters to Streamlit-friendly markdown math
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)

    # optional prose normalization
    text = text.replace("T_g", "Tg")

    return text.strip()
    
# ====================================================
# 🧠 🔥 [核心逻辑] 意图分析与坐标轴推断
# ====================================================
def _analyze_plot_intent(inputs: dict) -> dict:
    """
    分析用户的查询 inputs，决定：
    1. mode: "3D" (力学) 还是 "2D" (热学/其他)
    2. axes: (x, y, z) 或 (x, y)
    """
    mech_props = {"tensile_strength", "youngs_modulus", "elongation_at_break"}
    therm_props = {"glass_transition", "melting_temperature"}

    mentioned = []
    for key in inputs.keys():
        clean = key.replace("min_", "").replace("max_", "").replace("target_", "")
        if clean in mech_props or clean in therm_props:
            mentioned.append(clean)
    mentioned = list(set(mentioned))

    mech_hit = sum(1 for p in mentioned if p in mech_props)
    therm_hit = sum(1 for p in mentioned if p in therm_props)

    # 情况 A: 主要是力学 -> 画 3D (模量, 强度, 伸长率)
    if mech_hit > 0 or (mech_hit == 0 and therm_hit == 0):
        x, y, z = "youngs_modulus", "tensile_strength", "elongation_at_break"
        return {"mode": "3D", "reason": "Mechanics Dominant", "axes": (x, y, z)}

    # 情况 B: 主要是热学 -> 画 2D (Tm vs Tg)
    if therm_hit > 0:
        x, y = "melting_temperature", "glass_transition"
        return {"mode": "2D", "reason": "Thermal Dominant", "axes": (x, y)}

    # 兜底
    return {
        "mode": "3D",
        "reason": "Default",
        "axes": ("youngs_modulus", "tensile_strength", "elongation_at_break"),
    }
    
# ====================================================
# ⭐ Plan Executor 主函数
# ====================================================
def plan_executor(
    original_query: str,
    plan: dict | None,
    domain: dict | None,
    export_csv: bool = True,
    idea_prefix: str = "",
    memory=None,
    allow_save_snapshot: bool = True,
    verbose: bool = True,
    enable_quality_loop: bool = True,
    export_path: str | None = None,
    retrieval_mode: str = "hybrid",
    design_profile: dict | None = None,
) -> str:
    """
    [Plan Executor] 计划执行器
    核心职责：执行 Plan 中的每一步，并管理上下文和文件保存。
    """

    current_save_dir = export_path if export_path else STEP_LOG_DIR
    os.makedirs(current_save_dir, exist_ok=True)

    def try_save_memory(final_result_text):
        if memory and allow_save_snapshot:
            try:
                target_dir = current_save_dir
                if verbose:
                    console.print(f"[cyan]💾 正在调用 Memory 进行存档...[/cyan]")

                saved_path = memory.save_memory_snapshot(
                    folder_path=target_dir,
                    original_question=original_query,
                    final_answer=final_result_text,
                )
                if verbose:
                    console.print(f"[bold green]✔ Memory Snapshot 已保存: {saved_path}[/bold green]")
            except Exception as e:
                console.print(f"[red]⚠️ Memory 存档失败: {e}[/red]")

    # ==========================================================
    # 0) 初始化全局上下文（方便前端判断“有没有调用 table_query”）
    # ==========================================================
    execution_context = {
        "called_mech_table": False,   # ✅ 只要走过 query_mech_table 分支就置 True
        "idea_visuals": [],           # ✅ 每次 query_mech_table 都会 append 一条
    }
    last_response = "No execution result."

    # ==========================================================
    # 1) 兜底逻辑：没有 Plan -> 直接 RAG
    # ==========================================================
    if plan is None or "steps" not in plan:
        if verbose:
            console.print("[yellow]⚠️ Planner 返回为空，直接用原问题检索。[/yellow]")

        fallback_id = f"raw_query_{idea_prefix}" if idea_prefix else "raw_query"

        if enable_quality_loop and rag_answer_with_quality_loop:
            if verbose:
                console.print("[magenta]🔄 Fallback with Quality Loop...[/magenta]")
            try:
                final_rag_result = rag_answer_with_quality_loop(
                    original_query,
                    domain=domain,
                    top_k=15,
                    export_csv=export_csv,
                    step_id=fallback_id,
                    max_attempts=2,
                    export_path=current_save_dir,
                )
            except TypeError:
                final_rag_result = rag_answer_with_quality_loop(
                    original_query,
                    domain=domain,
                    top_k=15,
                    export_csv=export_csv,
                    step_id=fallback_id,
                    max_attempts=2,
                )
        else:
            try:
                final_rag_result = rag_answer(
                    original_query,
                    domain=domain,
                    top_k=15,
                    export_csv=export_csv,
                    step_id=fallback_id,
                    export_path=current_save_dir,
                    retrieval_mode=retrieval_mode,
                )
            except TypeError:
                final_rag_result = rag_answer(
                    original_query,
                    domain=domain,
                    top_k=15,
                    export_csv=export_csv,
                    step_id=fallback_id,
                    retrieval_mode=retrieval_mode,
                )

        try_save_memory(final_rag_result)

        # ✅ 兜底也落盘（让前端读取 execution_context.json 不报错）
        try:
            ctx_path = os.path.join(current_save_dir, "execution_context.json")
            with open(ctx_path, "w", encoding="utf-8") as f:
                json.dump(execution_context, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

        return final_rag_result

    # ==========================================================
    # 2) 正常执行 Plan
    # ==========================================================
    steps = plan["steps"]

    if verbose:
        console.print(f"\n🔧 [bold green]🚀 开始执行 Plan ({idea_prefix or 'Main'})...[/bold green]")
        console.print(f"[dim]📂 Results will be saved to: {current_save_dir}[/dim]")

    for idx, step in enumerate(steps):
        clean_sid = f"step{idx + 1}"
    
        if not isinstance(step, dict):
            msg = f"⚠️ Invalid step object at {clean_sid}: {repr(step)}"
            print(f"[DEBUG] {msg}", flush=True)
            console.print(f"[yellow]{msg}[/yellow]")
            execution_context[clean_sid] = msg
            last_response = msg
            continue
    
        original_sid_from_llm = step.get("id", "unknown_id")
        stype = step.get("type")
    
        inputs = step.get("inputs", {})
        if inputs is None:
            print(f"[DEBUG] {clean_sid} inputs is None, auto-fixing to {{}}", flush=True)
            inputs = {}
        elif not isinstance(inputs, dict):
            print(f"[DEBUG] {clean_sid} inputs is not dict: {type(inputs)} -> auto-fixing to {{}}", flush=True)
            inputs = {}

        current_step_id = f"{clean_sid}_{idea_prefix}" if idea_prefix else clean_sid
        display_prefix = f"[{idea_prefix}] " if idea_prefix else ""

        if verbose:
            console.rule(f"[bold blue]➡ {display_prefix}Running: {clean_sid} ({stype})[/bold blue]")

        # ------------------------------------------------------------------
        # ① query_mech_table
        # ------------------------------------------------------------------
        if stype == "query_mech_table":
            execution_context["called_mech_table"] = True  # ✅ 只要进入就算调用过

            # ✅ 先创建 visual_item（不依赖 result / MECH_DB_PATH / 画图是否成功）
            #   这样前端永远能知道“调用了 table_query，但为什么没图”
            visual_item = {
                "idea": idea_prefix or "Main",
                "step": clean_sid,
                "tag": current_step_id,
                "ashby": None,
                "pca": None,
                "n_records": 0,
                "db_path_ok": bool(MECH_DB_PATH),
                "reason": "",  # 可写入原因：no_records / no_db_path / plot_failed / ok
            }
            execution_context.setdefault("idea_visuals", [])
            execution_context["idea_visuals"].append(visual_item)

            if not run_mech_table_query:
                msg = "❌ 缺少 run_mech_table_query 模块，跳过此步骤。"
                console.print(f"[red]{msg}[/red]")
                execution_context[clean_sid] = msg
                execution_context[original_sid_from_llm] = msg
                last_response = msg
                visual_item["reason"] = "missing_run_mech_table_query"
                continue

            try:
                result = run_mech_table_query(inputs, max_rows=15)

                execution_context[clean_sid] = result
                execution_context[original_sid_from_llm] = result

                # 保存原始表结果
                save_step_result(current_step_id, stype, result, step_log_dir=current_save_dir, console=console)

                count = len(result) if isinstance(result, list) else 0
                visual_item["n_records"] = count

                if verbose:
                    console.print(f"[green]✔ 命中 {count} 条记录[/green]")
                last_response = f"Database query returned {count} records."

                # ✅ 没结果：也要明确写 reason
                if not result or count == 0:
                    visual_item["reason"] = "no_records"
                    continue

                # ✅ 没 DB 路径：也要明确写 reason
                if not MECH_DB_PATH:
                    visual_item["reason"] = "no_db_path"
                    continue

                # ==========================================================
                # 🔥 [画图] Ashby + PCA
                # ==========================================================
                plot_save_path = os.path.join(current_save_dir, "plots")
                os.makedirs(plot_save_path, exist_ok=True)

                plot_intent = _analyze_plot_intent(inputs)
                plot_mode = plot_intent["mode"]
                plot_axes = plot_intent["axes"]
                plot_reason = plot_intent["reason"]

                chart_path = None
                pca_path = None

                # A) Ashby
                try:
                    if plot_mode == "3D":
                        if verbose:
                            console.print(f"[dim]🧊 3D Ashby Mode ({plot_reason}): {plot_axes}[/dim]")
                        chart_path = plot_ashby_3d_chart(
                            target_results=result,
                            db_path=MECH_DB_PATH,
                            save_path=plot_save_path,
                            x_axis=plot_axes[0],
                            y_axis=plot_axes[1],
                            z_axis=plot_axes[2],
                            tag=current_step_id,
                        )
                    else:
                        if verbose:
                            console.print(f"[dim]📉 2D Ashby Mode ({plot_reason}): {plot_axes}[/dim]")
                        chart_path = plot_ashby_chart(
                            target_results=result,
                            db_path=MECH_DB_PATH,
                            save_path=plot_save_path,
                            x_axis=plot_axes[0],
                            y_axis=plot_axes[1],
                            tag=current_step_id,
                        )

                    if chart_path:
                        visual_item["ashby"] = chart_path  # ✅ 回填
                        visual_item["reason"] = "ok"
                        chart_name = os.path.basename(chart_path)

                        execution_context[f"{current_step_id}_chart"] = chart_path
                        execution_context[f"{clean_sid}_chart"] = chart_path

                        last_response += f"\n[VISUALIZATION] Ashby ({plot_mode}) generated: {chart_name}"

                except Exception as e:
                    console.print(f"[yellow]⚠️ Ashby Plotting failed: {e}[/yellow]")
                    if not visual_item["reason"]:
                        visual_item["reason"] = "ashby_plot_failed"

                # B) PCA
                try:
                    if verbose:
                        console.print("[dim]🧮 Calculating PCA for 5D material space...[/dim]")

                    pca_path = plot_pca_chart(
                        target_results=result,
                        db_path=MECH_DB_PATH,
                        save_path=plot_save_path,
                        tag=current_step_id,
                    )

                    if pca_path:
                        visual_item["pca"] = pca_path  # ✅ 回填
                        name = os.path.basename(pca_path)

                        execution_context[f"{current_step_id}_pca"] = pca_path
                        execution_context[f"{clean_sid}_pca"] = pca_path

                        last_response += f"\n[VISUALIZATION] PCA generated: {name}"
                        if visual_item["reason"] in ("", "ashby_plot_failed"):
                            visual_item["reason"] = "ok"

                except Exception as e:
                    console.print(f"[yellow]⚠️ PCA Plotting failed: {e}[/yellow]")
                    if visual_item["reason"] == "":
                        visual_item["reason"] = "pca_plot_failed"

            except Exception as e:
                console.print(f"[red]⚠️ 数据库查询出错: {e}[/red]")
                err_msg = f"Error: {e}"
                execution_context[clean_sid] = err_msg
                execution_context[original_sid_from_llm] = err_msg
                last_response = err_msg
                visual_item["reason"] = "query_failed"

            continue

        # ------------------------------------------------------------------
        # ② search_papers
        # ------------------------------------------------------------------
        if stype == "search_papers":
            q = inputs.get("query", "")
            k = 15
            inputs["top_k"] = 15

            step_mode = inputs.get("retrieval_mode", retrieval_mode)
            if step_mode not in ("hybrid", "dense", "bm25"):
                step_mode = retrieval_mode

            query_policy = inputs.get("query_policy", "unspecified")
            step_query_rewrite_mode = inputs.get("query_rewrite_mode")
            if step_query_rewrite_mode not in ("none", "domain_append"):
                step_query_rewrite_mode = "none" if query_policy == "preserve_original" else "domain_append"

            if verbose:
                console.print(f"[cyan]🔍 检索文献: {q}[/cyan]")
                console.print(f"[dim]🔧 retrieval_mode = {step_mode} | query_policy = {query_policy} | query_rewrite_mode = {step_query_rewrite_mode}[/dim]")
                if enable_quality_loop:
                    console.print("[yellow]ℹ️ Planner search_papers 已改为 retrieval-only；此处不再执行 quality loop。[/yellow]")
            try:
                print(f"[DEBUG] {clean_sid}: before retrieve_and_rerank_evidence", flush=True)
            
                retrieval_pack = retrieve_and_rerank_evidence(
                    query=q,
                    domain=domain,
                    top_k=k,
                    export_csv=export_csv,
                    step_id=current_step_id,
                    export_path=current_save_dir,
                    retrieval_mode=step_mode,
                    query_rewrite_mode=step_query_rewrite_mode,
                    use_rerank=True,
                )
            
                print(f"[DEBUG] {clean_sid}: after retrieve_and_rerank_evidence -> {type(retrieval_pack)}", flush=True)
                evidence_items = retrieval_pack.get("evidence_items", [])
                print(f"[DEBUG] {clean_sid}: evidence_items_len = {len(evidence_items) if isinstance(evidence_items, list) else 'NON_LIST'}", flush=True)
    
            except Exception as e:
                console.print(f"[red]Search Error: {e}[/red]")
                print(f"[DEBUG] {clean_sid} retrieve_and_rerank_evidence failed: {repr(e)}", flush=True)
                retrieval_pack = None
            
            # 🔥 关键补丁：防止函数“返回 None 但不抛异常”
            if retrieval_pack is None:
                print(f"[DEBUG] {clean_sid} retrieval_pack is None, fallback to {{}}", flush=True)
                retrieval_pack = {}
            elif not isinstance(retrieval_pack, dict):
                print(f"[DEBUG] {clean_sid} retrieval_pack is {type(retrieval_pack)}, fallback to {{}}", flush=True)
                retrieval_pack = {}
            
            evidence_items = retrieval_pack.get("evidence_items", [])
            if evidence_items is None:
                print(f"[DEBUG] {clean_sid} evidence_items is None, fallback to []", flush=True)
                evidence_items = []
            elif not isinstance(evidence_items, list):
                print(f"[DEBUG] {clean_sid} evidence_items is {type(evidence_items)}, fallback to []", flush=True)
                evidence_items = []
            
            step_payload = {
                "query": q,
                "enhanced_query": retrieval_pack.get("enhanced_query", q),
                "query_policy": query_policy,
                "query_rewrite_mode": retrieval_pack.get("query_rewrite_mode", step_query_rewrite_mode),
                "evidence_items": evidence_items,
                "mode": step_mode,
                "rag_answer": None,
            }

            execution_context[clean_sid] = step_payload
            execution_context[original_sid_from_llm] = step_payload

            n_hits = len(step_payload["evidence_items"])
            last_response = f"Retrieved and reranked {n_hits} evidence passages for: {q}"
            continue

        # ------------------------------------------------------------------
        # ③ reason_over_evidence
        # ------------------------------------------------------------------
        if stype == "reason_over_evidence":
            report_style = inputs.get("report_style", "experimental_manuscript")
            used = inputs.get("use_steps", [])
            
            # 如果 Planner 漏写 use_steps，自动回接所有已有上游步骤
            if not used:
                used = [f"step{i}" for i in range(1, idx + 1) if f"step{i}" in execution_context]
                if verbose:
                    console.print(f"[yellow]⚠️ use_steps 缺失，自动回接上游步骤: {used}[/yellow]")
            
            evd = {}
            for u in used:
                val = execution_context.get(
                    u,
                    execution_context.get(
                        f"{u}_{idea_prefix}",
                        execution_context.get(original_sid_from_llm, "No data"),
                    ),
                )
                evd[u] = val

            domain_str = json.dumps(domain or {}, ensure_ascii=False, indent=2)

            if not evd:
                if verbose:
                    console.print("[yellow]⚠️ 上游步骤无数据，使用空数据继续。[/yellow]")
                evd = {"error": "No upstream data found."}
            structured_evidence_text, reference_map_text, ordered_source_ids = _build_structured_evidence_text(evd)
            structured_table_text = _build_structured_table_text(evd)

            if not idea_prefix:
                role_definition = "You are a distinguished Professor of Degradable Polymer Science."
                task_instruction = (
                    f"Task: Synthesize the evidence and provide a comprehensive, scientific explanation for: '{original_query}'."
                )

                output_format = """
Output Structure:
1. **Comprehensive Overview**: Provide a direct but highly detailed academic overview of the topic.
2. **In-depth Molecular & Scientific Mechanisms**: Explain the fundamental 'Why' and 'How' in extreme detail. Break down the polymer physics, chemical interactions, and structural factors step-by-step. Elaborate extensively using professional scientific reasoning.
3. **Evidence Synthesis & Literature Trends**: Extensively synthesize the findings from the provided papers. Provide quantitative data, comparisons, and nuances discussed in the literature. Perform a deep critical analysis.
4. **Limitations & Uncertainties**: Discuss any boundaries, trade-offs, or uncertainties related to this scientific mechanism.
5. **References**
[[AUTO_REFERENCES]]
"""
                qa_format_rule = """
⚠️ FORMAT RULES FOR QA:
- Output plain scientific English.
- Do NOT use HTML tags such as <sub>, <sup>, <i>, <b>, <br>, or similar markup.
- You may include an equation ONLY if it is explicitly supported by the retrieved evidence and is necessary to answer the question.
- If you include an equation, use ONLY standard Markdown math delimiters:
  - inline math: $...$
  - display math: $$...$$
- Do NOT use \\[...\\] or \\(...\\).
- After any equation, define variables in plain English, for example:
  - w_LA = weight fraction of lactic acid units
  - w_GA = weight fraction of glycolic acid units
  - Tg,LA = glass transition temperature of the LA-rich component
  - Tg,PGA = glass transition temperature of PGA
- Do NOT use broken symbolic prose such as "where (w_{LA}) and (w_{GA})".
- Do NOT introduce textbook-style equations unless they are directly supported by the retrieved evidence.
"""

                prompt = f"""
{role_definition}

Original User Query: {original_query}
Context Focus: General Scientific Inquiry
Domain info:
{domain_str}

Reference Map (SOURCE IDENTITY MAP ONLY):
{reference_map_text}

Structured Literature Evidence:
{structured_evidence_text}

Structured Table / Database Evidence:
{structured_table_text}

{task_instruction}

{output_format}

{qa_format_rule}

⚠️ **CITATION FORMATTING RULES (STRICT)**:
The Reference Map above is provided only to identify the correct cleaned literature source IDs and deduplicate repeated sources.

1. Use [n] citations only for literature-supported claims grounded in the Structured Literature Evidence.
2. If you use structured database/table evidence, describe it explicitly as database/table evidence, but do NOT fabricate literature-style source IDs for it.
3. In the main body, cite specific mechanisms, quantitative claims, processing conditions, and comparisons from literature using [n].
4. In the References section, DO NOT generate any reference entries yourself.
5. Keep the exact placeholder `[[AUTO_REFERENCES]]` unchanged in the References section.
6. Do not cite unsupported sources or create citations that are not grounded in the provided evidence.
7. DO NOT output raw filenames, chunk suffixes, or file extensions.
8. If multiple evidence snippets come from the same source, keep them under the same [n].
"""

            if verbose:
                console.print(
                    f"[magenta]📝 LLM 正在进行深度推理与最终生成 ({'Validation' if idea_prefix else 'QA'})...[/magenta]"
                )
            if idea_prefix:
                report_category = classify_design_report_category(
                    original_query=original_query,
                    idea_prefix=idea_prefix,
                    domain=domain or {},
                    structured_evidence_text=structured_evidence_text,
                    structured_table_text=structured_table_text,
                    preset_profile=design_profile,
                )
            
                if report_category is None:
                    print(f"[DEBUG] report_category is None at {clean_sid}, fallback to {{}}", flush=True)
                    report_category = {}
                elif not isinstance(report_category, dict):
                    print(f"[DEBUG] report_category is not dict: {type(report_category)} -> fallback to {{}}", flush=True)
                    report_category = {}
            
                execution_context[f"{clean_sid}_report_category"] = report_category
                execution_context[f"{original_sid_from_llm}_report_category"] = report_category
                execution_context["final_design_profile"] = report_category
                execution_context["final_design_report_category"] = report_category

                print(f"[DEBUG] ordered_source_ids count = {len(ordered_source_ids)}", flush=True)
                print(f"[DEBUG] reference_map_text =\n{reference_map_text}", flush=True)
                print(f"[DEBUG] structured_evidence_text head = {structured_evidence_text[:1000]}", flush=True)
                
                reasoning = generate_design_report_from_structured_evidence(
                    original_query=original_query,
                    idea_prefix=idea_prefix,
                    domain=domain or {},
                    structured_evidence_text=structured_evidence_text,
                    structured_table_text=structured_table_text,
                    reference_map_text=reference_map_text,
                    ordered_source_ids=ordered_source_ids,
                    report_category=report_category,
                    llm_callable=call_deepseek_llm,
                )

                # 🔥 设计报告也必须走和 QA 一样的后处理
                reasoning = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()

                # 去掉模型可能输出的 markdown 代码围栏
                reasoning = re.sub(r"^\s*```(?:markdown)?\s*", "", reasoning, flags=re.IGNORECASE)
                reasoning = re.sub(r"\s*```\s*$", "", reasoning)

                # 统一按正文首次出现顺序重建引用与 References
                reasoning = _normalize_citations_and_rebuild_references(reasoning, ordered_source_ids)

                # 清理格式
                reasoning = _clean_query_formatting(reasoning)
            else:
                raw_response = call_deepseek_llm(prompt)

                reasoning = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
                if not reasoning:
                    if verbose:
                        console.print(
                            "[yellow]⚠️ Warning: Response became empty after cleaning <think> tags. Reverting to raw response.[/yellow]"
                        )
                    reasoning = raw_response

                # 🔧 后处理：按正文首次出现顺序统一重排引用编号
                reasoning = _normalize_citations_and_rebuild_references(reasoning, ordered_source_ids)
                reasoning = _clean_query_formatting(reasoning)
            
            execution_context[clean_sid] = reasoning
            execution_context[original_sid_from_llm] = reasoning

            save_step_result(
                current_step_id,
                stype,
                reasoning,
                step_log_dir=current_save_dir,
                console=console,
                suffix="result",
            )

            last_response = reasoning
            try_save_memory(reasoning)
            continue

        # ------------------------------------------------------------------
        # ④ 未知 step type：记录一下，避免 silently skip
        # ------------------------------------------------------------------
        msg = f"⚠️ Unknown step type: {stype}"
        console.print(f"[yellow]{msg}[/yellow]")
        execution_context[clean_sid] = msg
        execution_context[original_sid_from_llm] = msg
        last_response = msg
        continue

    # ==========================================================
    # ✅✅✅ for 循环结束后：统一保存 execution_context.json
    # ==========================================================
    try:
        ctx_path = os.path.join(current_save_dir, "execution_context.json")
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(execution_context, f, ensure_ascii=False, indent=2, default=str)
        if verbose:
            console.print(f"[dim]🧾 execution_context saved: {ctx_path}[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to save execution_context.json: {e}[/yellow]")

    return last_response
