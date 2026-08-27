# Agent/research_agent.py
import re
import os
import traceback
import concurrent.futures
from rich.console import Console
from rich.table import Table

# ====================================================
# 📦 Module Imports
# ====================================================
from Agent.Agent_Config.agent_config import console, STEP_LOG_DIR
from Agent.Agent_Config.deepseek_client import call_deepseek_llm
from Agent.RAG import vector_store
from Agent.RAG.vector_store import load_vector_store
from Agent.Intent.domain_router import classify_domain
from Agent.Intent.router_module import answer_with_router
# ✅ Import NoStreamlitLog for safe threading
from Agent.Utils.file_utils import save_step_result, NoStreamlitLog 

# 导入新拆分的 Worker 和 Utils
from Agent.Planner.pipeline_worker import process_single_idea_pipeline
from Agent.Planner.planner_module import run_planner
from Agent.Planner.plan_executor import plan_executor

# 引入可视化和doc工具 (带容错)
try:
    from Agent.Utils.doc_converter import convert_md_to_doc
    from Agent.Utils.data_visualizer import create_comparison_chart
    from Agent.Utils.idea_visualizer import plot_initial_ranking, plot_final_radar
except ImportError:
    print("⚠️ Warning: Utils modules (doc/chart) not found. Skipping visualization.")
    convert_md_to_doc = None
    create_comparison_chart = None

# 尝试导入 Memory 模块
try:
    from Agent.Memory.memory import Memory
except ImportError:
    try:
        from Agent.memory.memory import Memory
    except ImportError:
        print("⚠️ Warning: Could not import Memory module. Running without memory.")
        Memory = None

if not console:
    console = Console()

# ====================================================
# 🤖 ResearchAgent Class
# ====================================================
class ResearchAgent:
    def __init__(self, use_memory=True, enable_short_term=True, enable_long_term=True, max_short_k=3, max_long_k=2):
        """
        初始化 Agent
        """
        self.console = Console()
        self.use_memory = use_memory
        
        # 1. 加载向量库
        try:
            vector_store_ready = (
                vector_store.metadata is not None
                and getattr(vector_store, "shards", None)
                and len(vector_store.shards) > 0
                and vector_store.bm25_model is not None
            )

            if vector_store_ready:
                self.console.print("[green]✅ Vector store already in memory. Skip reload.[/green]")
            else:
                load_vector_store()
                self.console.print("[green]✅ Vector store loaded successfully.[/green]")

        except Exception as e:
            self.console.print(f"[red]❌ Failed to load vector store: [/red]{e}")

        # 2. 初始化 Memory
        self.memory = None
        if use_memory and Memory:
            self.memory = Memory(
                max_short_term_size=max_short_k,
                use_short_term_memory=enable_short_term,
                max_long_term_size=max_long_k,
                use_long_term_memory=enable_long_term,
                default_unit="MPa",
                optimization_mode="max"
            )
            self.console.print(f"[bold green]🧠 Memory System Online[/bold green]")
        else:
            self.console.print("[yellow]⚪ Memory Module is DISABLED.[/yellow]")
            
        self.console.print(f"[dim]📂 Base Log Dir: {STEP_LOG_DIR}[/dim]")

    # 3. 运行主逻辑
    def run_one_step(self, query: str, enable_quality_loop: bool = True, use_short_term: bool = True, use_long_term: bool = True, save_path: str = None) -> str:
        """
        核心逻辑：Router -> Pipeline (Plan+Execute+Report)
        """
        # 1. 动态配置 Memory
        if self.memory:
            self.memory.use_short_term_memory = use_short_term
            self.memory.use_long_term_memory = use_long_term

        memory_is_active = (use_short_term or use_long_term) and (self.memory is not None)
        current_active_memory = self.memory if memory_is_active else None

        # 3. 路径处理
        # 🔥 [关键修复 5] 强制使用清洗后的路径，防止尾部空格导致保存失败
        current_export_path = save_path if save_path else STEP_LOG_DIR
        current_export_path = os.path.normpath(current_export_path) 
        os.makedirs(current_export_path, exist_ok=True)

        # Debug Log
        print("DEBUG: --------------------------------------------------")
        print(f"DEBUG: [System] 成功进入 run_one_step")
        print(f"DEBUG: [System] 收到用户指令: {query}")
        print(f"DEBUG: [Config] Output Dir: {current_export_path}")
        print("DEBUG: --------------------------------------------------")

        # --- 特殊指令 ---
        if query.lower() in ("new", "/clear", "reset"):
            if self.memory and hasattr(self.memory, 'clear'):
                self.memory.clear()
                msg = "🧹 Memory Cleared! Starting a fresh context."
                self.console.print(f"[bold green]{msg}[/bold green]")
                return msg
            else:
                return "❌ Memory object disabled."

        # --- 核心流程 ---
        try:
            print("DEBUG: [Router] 正在调用 Router...")
            self.console.print("\n🧩 [bold cyan]Step 0: Router + Design (Generating Ideas)...[/bold cyan]")
            
            # 🔥 关键：传入清洗后的 save_path 给 Router
            router_result = answer_with_router(query, verbose=True, save_path=current_export_path)
            
            route_type = router_result.get("route_type", "unknown")
            top_ideas = router_result.get("top_ideas", []) 
            all_ideas = router_result.get("all_ideas", [])

            # [新增逻辑] 绘制发散阶段的柱状图
            if route_type == "design" and top_ideas and plot_initial_ranking:
                self.console.print("[dim]📊 Generating initial screening chart for 8 ideas...[/dim]")
                # 注意：此时 router 已经把所有生成的想法放在了结果里，
                # 我们调用新脚本画出它们的初步评分对比
                plot_initial_ranking(top_ideas, current_export_path)
    
            default_plan = router_result.get("plan", None) 
            domain = router_result.get("domain", None)

            self.console.print(f"[bold magenta]🧭 Route type:[/bold magenta] {route_type}")

            # ========================
            # Branch A: Chat Mode
            # ========================
            if route_type == "chat":
                self.console.print("[yellow]💬 Chat Mode.[/yellow]")
                
                # 1. Get raw response
                raw_resp = call_deepseek_llm(query, system_prompt="You are BEAVER, a helpful assistant for Degradable Polymer Research.")
                
                # 2. 🔥 [New] Clean <think> tags
                # Use regex to remove thinking process
                resp = re.sub(r"<think>.*?</think>", "", raw_resp, flags=re.DOTALL).strip()
                
                # Fallback: if cleaning results in empty string, use raw response
                if not resp:
                    resp = raw_resp
                
                self.console.print(resp)
                return resp

            # 补充 Domain
            if domain is None:
                domain = classify_domain(query)

            # ========================
            # Branch B: Design Mode (并行流水线)
            # ========================
            if route_type == "design" and top_ideas:
                msg = f"🚀 Detected Design Mode. Pipelining {len(top_ideas)} ideas (Isolated Execution)..."
                self.console.print(f"\n[bold cyan]{msg}[/bold cyan]")
                
                ideas_for_plotting = all_ideas if all_ideas else top_ideas
                
                if plot_initial_ranking and ideas_for_plotting:
                    self.console.print(f"[dim]📊 Generating initial screening chart for {len(ideas_for_plotting)} ideas...[/dim]")
                    plot_initial_ranking(ideas_for_plotting, current_export_path)

                final_results_summary = []
                future_to_idea = {}
                tasks = []

                self.console.print("DEBUG: [Parallel] 正在提交线程任务...")
                
                # 准备任务参数
                for idx, idea in enumerate(top_ideas):
                    tasks.append((
                        (idx, idea), 
                        query, 
                        domain, 
                        None,   # 禁用 Memory
                        True,   # verbose
                        enable_quality_loop, 
                        current_export_path 
                    ))

                # --- 2. 执行并行任务并收集结果 ---
                
                # 🔥🔥🔥 [Critical Fix] Redirect logs to background terminal to prevent Streamlit UI crash 🔥🔥🔥
                self.console.print("⚠️ [System] Switching logs to Background Terminal to prevent UI Crash...")
                
                with NoStreamlitLog(self): 
                    # Dynamic concurrency
                    actual_workers = min(len(top_ideas), 3)
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
                        # A. 提交任务
                        for t in tasks:
                            ft = executor.submit(process_single_idea_pipeline, t)
                            future_to_idea[ft] = t[0]

                        # B. 收集结果
                        for future in concurrent.futures.as_completed(future_to_idea):
                            idea_idx, idea_info = future_to_idea[future]
                            try:
                                data = future.result()
                                if data:
                                    final_results_summary.append(data)
                                    # Use safe print (goes to background terminal due to context manager)
                                    print(f"✔ Pipeline Finished: {data['id']} - {data['title']}")
                                    
                                    # 写入 Memory (线程安全操作)
                                    if self.memory and memory_is_active:
                                        self.memory.store(
                                            next_guess={
                                                "composition": data['title'],
                                                "reason": data['mechanism'],
                                                "reflection": f"Verified {data['id']}"
                                            },
                                            feedback=str(data['result'])[:300],
                                            predicted_value=0.0
                                        )
                            except Exception as exc:
                                err_msg = f"❌ Pipeline failed for idea{idea_idx+1}: {exc}"
                                print(err_msg, flush=True)
                                traceback.print_exc()
                                # 失败兜底，保证列表长度和ID对齐，避免绘图报错
                                final_results_summary.append({
                                    "id": f"idea{idea_idx+1}",
                                    "title": idea_info.get("idea_name", "Failed Idea"),
                                    "mechanism": "Execution Error",
                                    "result": f"Error: {exc}",
                                    "report_summary": "Analysis failed due to execution error.",
                                    "score": 0,
                                    "scores_dict": {
                                        "Feasibility": 0,
                                        "Predictability": 0,
                                        "Performance": 0,
                                        "Innovation": 0,
                                        "Chemical Validity": 0,
                                    },
                                    "chemical_validity": {
                                        "overall_chemical_validity": 0,
                                        "contradiction_detected": True,
                                        "major_red_flags": ["Pipeline failed before Chemical Validity check."],
                                        "critic_status": "pipeline_failed",
                                    }
                                })

                # 🔥 Log switching back to UI
                self.console.print(f"[green]✅ All {len(final_results_summary)} parallel tasks completed![/green]")

                # --- 3. 汇总与 UI 展示 ---
                if final_results_summary:
                    final_results_summary.sort(key=lambda x: x['id'])
                    
                    # --- [新增] 收敛阶段可视化 (Final Radar) ---
                    if plot_final_radar:
                        self.console.print("[bold green]📊 Generating final radar chart for Top 3 ideas...[/bold green]")
                        plot_final_radar(final_results_summary, current_export_path)
                    
                    # 打印表格
                    self._print_summary_table(final_results_summary)

                    # 生成旧版对比图 (可选，保留作为补充)
                    chart_path = None
                    if create_comparison_chart:
                        self.console.print("DEBUG: [Chart] 正在生成对比图表 (Local 3D)...")
                        chart_path = create_comparison_chart(final_results_summary, current_export_path)
                    
                    # 生成报告 (传入两张新图的引用)
                    ui_report = self._generate_ui_report(query, final_results_summary, chart_path)

                    # 保存 Snapshot
                    if self.memory and memory_is_active:
                        self.memory.save_memory_snapshot(
                            folder_path=current_export_path, 
                            original_question=query, 
                            final_answer=ui_report 
                        )
                        self.console.print("DEBUG: [Memory] Snapshot 已保存。")
                    
                    self.console.print("DEBUG: [Done] Design Mode 流程结束。")
                    return ui_report 
                else:
                    return "⚠️ No results generated."
            # ========================
            # Branch C: Standard Question Mode
            # ========================
            else: 
                self.console.print("[bold cyan]❓ Standard Question Mode.[/bold cyan]")
                
                if not default_plan:
                    default_plan = run_planner(
                        query, 
                        memory=current_active_memory, 
                        use_memory=memory_is_active, 
                        save_plan=False
                    )
                
                # 手动保存 Plan Step 1
                save_step_result("step1_plan", "plan", default_plan, current_export_path, console=self.console)

                # 执行计划 (传路径)
                final_answer = plan_executor(
                    query, 
                    default_plan, 
                    domain, 
                    export_csv=True, 
                    idea_prefix="",
                    memory=current_active_memory, 
                    allow_save_snapshot=False,        
                    enable_quality_loop=enable_quality_loop,
                    export_path=current_export_path  # 🔥 传递路径
                )

                # 写入 Memory & Snapshot
                if self.memory and memory_is_active:
                    self.memory.store(
                        next_guess={
                            "composition": "Standard Query",
                            "reason": "Direct user question",
                            "reflection": "N/A"
                        },
                        feedback=str(final_answer)[:300],
                        predicted_value=0.0
                    )
                    self.memory.save_memory_snapshot(
                        folder_path=current_export_path, 
                        original_question=query, 
                        final_answer=final_answer
                    )
                    self.console.print("[dim]💾 Standard query result added to memory.[/dim]")
                else:
                    self.console.print("[dim]⚪ Memory storage skipped (disabled).[/dim]")
                
                print("DEBUG: [Done] Standard Mode 流程结束。")

                self.console.file.flush()
                
                return str(final_answer)

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.console.print(f"[red]⚠️ Error: [/red]{e}\n{err_msg}")
            return f"❌ 系统运行出错: {e}\n\n查看后台控制台获取详细报错信息。"

    # --- 辅助方法：打印表格 ---
    def _print_summary_table(self, results):
        try:
            summary_table = Table(title="💡 Design Verification Summary")
            summary_table.add_column("ID", justify="center", style="cyan")
            summary_table.add_column("Idea Name", style="magenta")
            summary_table.add_column("Status", overflow="fold")
            summary_table.add_column("Score", justify="right", style="green")
            
            for item in results:
                has_result = bool(str(item.get("result", "")).strip())
                has_score = item.get("score", 0) > 0
            
                if has_result and has_score:
                    status_str = "✅ Report + Score Parsed"
                elif has_result:
                    status_str = "⚠️ Report Generated, Score Parse Failed"
                else:
                    status_str = "❌ Report Failed"
            
                summary_table.add_row(
                    item["id"],
                    item["title"],
                    status_str,
                    str(item.get("score", 0))
                )
            self.console.print(summary_table)
        except Exception:
            pass

    # --- 辅助方法：生成 Markdown 报告 ---
    def _generate_ui_report(self, query, results, chart_path):
        ui_report = f"### 🧪 Design Verification Results\n\n"
        ui_report += f"**User Query:** {query}\n\n"
        # [新增] 插入发散阶段的初筛图
        ui_report += "#### 1. Initial Screening (Divergence Phase)\n"
        for item in results:
            summary = item.get("report_summary", "No summary extracted.")
            s_dict = item.get("scores_dict", {})
            score_display = (
                f" (Avg: {item.get('score', 0)}) | "
                f"Feas:{s_dict.get('Feasibility',0)} "
                f"Pred:{s_dict.get('Predictability',0)} "
                f"Perf:{s_dict.get('Performance',0)} "
                f"Inno:{s_dict.get('Innovation',0)} "
                f"Chem:{s_dict.get('Chemical Validity',0)}"
            )
            
            ui_report += f"---\n"
            ui_report += f"### 📄 {item['title']} ({item['id']}){score_display}\n"
            ui_report += f"**💡 Core Mechanism**: {item['mechanism']}\n\n"
            ui_report += f"> **📖 Abstract**\n> \n"
            ui_report += f"> {summary}\n\n"
            ui_report += f"✅ *Full Report & Word Doc generated locally.*\n"
            ui_report += f"---\n"
        return ui_report