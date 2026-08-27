# -*- coding: utf-8 -*-
import os
import sys
import glob
import time
import json
import httpx
import traceback
import socket  # 🔥 [新增] 用于设置全局超时
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')

original_httpx_init = httpx.Client.__init__
def patched_httpx_init(self, *args, **kwargs):
    if 'timeout' not in kwargs or kwargs['timeout'] is httpx.USE_CLIENT_DEFAULT:
        kwargs['timeout'] = httpx.Timeout(900.0) # 强制 15 分钟 (900秒) 超时
    original_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_httpx_init

from modules.utils import StreamToStatus

# Try to import UI components
try:
    from ui_components import render_download_section
except ImportError:
    def render_download_section(path):
        st.warning("⚠️ ui_components.py not found.")


# =======================================================
# 🧰 Small Utils
# =======================================================
def _find_latest_file(base_dir: str, patterns: list[str]) -> str | None:
    """
    Recursively search for files matching patterns in base_dir and return the latest one.
    patterns: ["**/xxx.png", "**/*.json"]
    """
    if not base_dir or not os.path.exists(base_dir):
        return None

    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(base_dir, pat), recursive=True))

    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _load_execution_context(run_dir: str) -> dict:
    """
    🔥 [Revised] Recursively find ALL execution_context.json files and merge idea_visuals
    """
    merged_context = {
        "idea_visuals": []
    }
    
    try:
        if not run_dir or not os.path.exists(run_dir):
            return {}
            
        # Find all matching json files
        json_files = glob.glob(os.path.join(run_dir, "**", "execution_context.json"), recursive=True)
        
        # Sort by time
        json_files.sort(key=os.path.getmtime)
        
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Merge visuals
                    if "idea_visuals" in data and isinstance(data["idea_visuals"], list):
                        merged_context["idea_visuals"].extend(data["idea_visuals"])
                        
            except Exception:
                continue
                
        return merged_context
        
    except Exception:
        return {}

def _tail_file(path: str, max_chars: int = 5000) -> str:
    """读取文件末尾字符（内存安全版）"""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, 2)  # 移动到文件末尾
            size = f.tell()
            # 只读取最后 max_chars 个字符
            f.seek(max(size - max_chars, 0))
            return f.read()
    except Exception:
        return ""

def _render_log_expander(log_path: str, title: str = "🪵 查看实时日志（从文件恢复）", max_lines: int = 200):
    """
    在 UI 里显示落盘日志尾部
    🔥 [修复] max_lines 转换为 max_chars，防止 TypeError
    """
    if not log_path or not os.path.exists(log_path):
        return
    with st.expander(title, expanded=False):
        # 估算字符数：假设平均每行 200 字符
        estimated_chars = max_lines * 200
        
        # 使用 max_chars 参数调用
        text = _tail_file(log_path, max_chars=estimated_chars)
        
        if text.strip():
            st.code(text, language="text")
        else:
            st.info("日志文件存在，但目前为空。")

def _log_recently_updated(log_path: str, window_seconds: int = 45) -> bool:
    """
    用日志文件更新时间判断后台任务是否可能仍在继续。
    只要日志在最近 window_seconds 秒内还在更新，就先不要判定为中断。
    """
    if not log_path or not os.path.exists(log_path):
        return False
    try:
        age = time.time() - os.path.getmtime(log_path)
        return age <= window_seconds
    except Exception:
        return False

def _render_idea_visuals(idea_visuals: list, title: str = "📊 Idea Visualizations"):
    """
    Render execution_context['idea_visuals'] into UI cards.
    """
    if not idea_visuals:
        return

    st.markdown(f"### {title}")

    for item in idea_visuals:
        idea = item.get("idea", "Main")
        step = item.get("step", "")
        # tag = item.get("tag", "") # Unused currently
        nrec = item.get("n_records", 0)
        ashby = item.get("ashby", None)
        pca = item.get("pca", None)
        reason = item.get("reason", "")

        # Header: Display more info in the expander title
        header = f"🔹 {idea} | {step}"
        
        # Status Icon & State
        if nrec > 0:
            header += f" | ✅ Found {nrec} records"
            expanded_state = True
        else:
            header += f" | ⚠️ No data found (0 hits)"
            expanded_state = False

        with st.expander(header, expanded=expanded_state):
            # Case A: Data found (n > 0)
            if nrec > 0:
                cols = st.columns(2)
                
                # Ashby Column
                with cols[0]:
                    if ashby and os.path.exists(ashby):
                        st.caption("📉 **Material Property Space (Ashby)**")
                        st.image(ashby, width="stretch")
                    else:
                        st.warning("⚠️ Ashby Chart generation failed (likely insufficient data dimensions)")

                # PCA Column
                with cols[1]:
                    if pca and os.path.exists(pca):
                        st.caption("🧮 **High-Dim Analysis (PCA)**")
                        st.image(pca, width="stretch")
                    else:
                        st.warning("⚠️ PCA Chart generation failed (likely missing key properties)")

            # Case B: No Data (n = 0)
            else:
                if reason == "no_db_path":
                    st.info("ℹ️ Material database path is not configured. Visualization analysis cannot be performed.")
                elif reason == "no_records":
                    st.info("ℹ️ No existing materials matching the design parameters were found in the database.")
                else:
                    st.info(f"ℹ️ No visualization charts generated for this step. Reason: {reason or 'No records found'}")


# =======================================================
# 📊 Radar Chart Function
# =======================================================
def render_four_dim_radar(scores_dict):
    """Draw 4-dim radar: Feasibility, Predictability, Performance, Innovation"""
    categories = ['Feasibility', 'Predictability', 'Performance', 'Innovation']

    values = [
        scores_dict.get('Feasibility', 0),
        scores_dict.get('Predictability', 0),
        scores_dict.get('Performance', 0),
        scores_dict.get('Innovation', 0)
    ]
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Scores',
        line_color='#0068c9',
        fill_color='rgba(0, 104, 201, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color='gray')),
            angularaxis=dict(tickfont=dict(size=14, color='black'), rotation=90)
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def render_chat_interface(usage_manager):
    """Main chat interface rendering function with Auto-Recovery and Run-Token protection"""

    # ============================================================
    # 🔥 [A-0] 灾难恢复逻辑：检测异常刷新并恢复上下文
    # ============================================================
    # 如果 session 中的 last_run_dir 丢失（说明页面刷新了），尝试从硬盘恢复
    if False:
        try:
            # 1. 尝试定位 Session_Runs 目录
            # 优先使用 session 中的配置
            base_search_path = st.session_state.get("session_run_dir")
            
            # 如果 session 也没了，使用 Agent 目录下的统一会话输出目录
            if not base_search_path:
                 base_search_path = os.path.abspath(
                     os.path.join(os.path.dirname(__file__), "..", "..", "Session_Runs")
                 )
            
            # 如果备用路径也不存在，尝试当前目录下的默认文件夹
            if not os.path.exists(base_search_path):
                 base_search_path = os.path.abspath(
                     os.path.join(os.path.dirname(__file__), "..", "..", "Session_Runs")
                 )

            # 2. 扫描并恢复
            if os.path.exists(base_search_path):
                # 找所有 Run_ 开头的文件夹
                all_runs = glob.glob(os.path.join(base_search_path, "Run_*"))
                if all_runs:
                    # 找最新的文件夹
                    latest_run = max(all_runs, key=os.path.getmtime)
                    
                    # 仅恢复 10 分钟内活跃的任务 (避免加载太久以前的)
                    if time.time() - os.path.getmtime(latest_run) < 600:
                        st.session_state.last_run_dir = latest_run
                        st.session_state.last_log_path = os.path.join(latest_run, "streamlit_live.log")
                    
                        # 恢复“正在运行”状态，让后面的无令牌重跑分支有机会先检查日志是否还活着
                        st.session_state.processing = True
                        st.session_state.current_processing_query = "__RECOVERING__"
                        st.session_state.run_token = False
                    
                        # 初始化消息列表，防止后续报错
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                    
                        st.toast(f"🔄 检测到页面刷新，已恢复上下文: {os.path.basename(latest_run)}", icon="🔌")
        except Exception as e:
            print(f"[Chat] Recovery check failed: {e}")

    # ============================================================
    # [A] Init State Locks
    # ============================================================
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_processing_query" not in st.session_state:
        st.session_state.current_processing_query = ""
    if "last_run_dir" not in st.session_state:
        st.session_state.last_run_dir = None
    if "messages" not in st.session_state:
        st.session_state.messages = [] 

    # ============================================================
    # [B] Render History Messages
    # ============================================================
    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "assistant")):

            # Phase 1: Divergence
            bar_path = msg.get("bar_chart")
            if bar_path and os.path.exists(bar_path):
                st.markdown("### 📉 Phase 1: Divergence Analysis")
                st.image(bar_path, caption="📉 Initial Screening: Divergence Analysis", width="stretch")

            # Text Content
            st.markdown(msg.get("content", ""))
            
            # ✅ Persisted Live Log (日志恢复显示)
            log_path = msg.get("log_path")
            if log_path and os.path.exists(log_path):
                _render_log_expander(log_path, title="🪵 查看本次运行日志（可恢复）", max_lines=300)

            # Idea visuals
            if "idea_visuals" in msg and msg["idea_visuals"] is not None:
                _render_idea_visuals(msg.get("idea_visuals", []), title="📊 Idea Visualizations")
            else:
                # Legacy compatibility
                ashby = msg.get("ashby_chart")
                pca = msg.get("pca_chart")
                if ashby and os.path.exists(ashby):
                    st.markdown("### 📊 Material Selection Map")
                    st.image(ashby, caption="📊 Ashby Plot: Material Property Space", width="stretch")
                if pca and os.path.exists(pca):
                    st.markdown("### 🧮 Data Reduction Map")
                    st.image(pca, caption="🧮 PCA: 5D to 2D Material Space", width="stretch")

            # Phase 2: Radar
            radar_path = msg.get("radar_chart")
            if radar_path and os.path.exists(radar_path):
                st.markdown("### 🧬 Innovation Assessment")
                st.image(radar_path, caption="🧬 Final Verification: Multi-Dimensional Evaluation", width="stretch")
            elif msg.get("scores"):
                st.markdown("### 🧬 Innovation Assessment")
                fig = render_four_dim_radar(msg["scores"])
                st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

            # Download Section
            dir_path = msg.get("dir_path")
            if dir_path and os.path.exists(dir_path):
                st.divider()
                render_download_section(dir_path)

    # ============================================================
    # [C] Core Interaction Logic
    # ============================================================
    input_placeholder = "🧠 Agent is thinking..." if st.session_state.processing else \
        "Enter your research query (e.g., Design a biodegradable polyester)..."

    # 1. 获取用户输入
    user_input = st.chat_input(
        placeholder=input_placeholder,
        key="main_chat_input",
        disabled=st.session_state.processing
    )

    # 2. 处理提交逻辑
    if user_input:
        # 先清掉上一轮的伪恢复痕迹
        st.session_state.last_log_path = None
        st.session_state.last_run_dir = None   # 如果你不需要跨轮恢复，建议也清
        st.session_state.run_token = True
        st.session_state.processing = True
        st.session_state.current_processing_query = user_input
        st.rerun()

    # 3. 执行任务逻辑 (带令牌检查)
    if st.session_state.processing and st.session_state.current_processing_query:
        
        # 检查是否有令牌 (Token)
        if st.session_state.get("run_token", False):
            # ✅ 合法启动 (持有令牌)
            
            # 🔥🔥 立即销毁令牌，防止后续刷新导致的无限重启
            st.session_state.run_token = False 
            
            st.toast("🚀 已开始执行任务", icon="🦫")
            
            try:
                process_agent_task(usage_manager)
            except Exception as e:
                st.error(f"Execution Error: {e}")
                # 记录错误
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 任务运行出错: {str(e)}"
                })
            finally:
                # 任务完成，状态复位并刷新
                st.session_state.processing = False
                st.rerun()

        else:
            # 🛑 场景：无令牌的重跑（更谨慎的恢复逻辑）
            if user_input is None:
                active_run_dir = st.session_state.get("last_run_dir")
                active_log = st.session_state.get("last_log_path")
        
                has_active_run_dir = bool(active_run_dir and os.path.exists(active_run_dir))
                log_is_alive = _log_recently_updated(active_log, window_seconds=45)
        
                # 先判断：是不是只是前端断了，但日志还在更新
                if st.session_state.processing and has_active_run_dir and log_is_alive:
                    st.toast("🔄 检测到连接抖动，后台任务可能仍在继续，已恢复日志视图。", icon="🔌")
                
                    if active_log and os.path.exists(active_log):
                        with st.expander("🪵 后台日志（恢复显示）", expanded=True):
                            st.code(_tail_file(active_log, max_chars=12000), language="text")
                
                    return
        
                # 只有日志长时间不再更新，才真正按“中断”处理
                if st.session_state.processing and has_active_run_dir:
                    log_content = "（无日志记录）"
                    if active_log and os.path.exists(active_log):
                        try:
                            log_content = _tail_file(active_log, max_chars=3000)
                        except Exception:
                            log_content = "（日志文件被占用，无法读取）"
        
                    bar_chart = _find_latest_file(active_run_dir, ["**/initial_screening_bar.png"])
                    radar_chart = _find_latest_file(active_run_dir, ["**/final_verification_radar.png"])
        
                    last_msg = st.session_state.messages[-1] if st.session_state.messages else {}
                    if "任务执行异常中断" not in last_msg.get("content", ""):
                        err_msg = {
                            "role": "assistant",
                            "content": f"""
        ⚠️ **任务执行异常中断 (Task Interrupted)**
        
        检测到本次任务的日志在一段时间内未继续更新，系统将其判定为已中断。
        为了防止数据丢失，已为您保存截至中断前的运行日志。您可以重新提交该任务。
        
        ---
        **📋 中断前的最后现场 (Last Logs):**
        ```text
        {log_content}
        ```""",
                            "log_path": active_log,
                            "dir_path": active_run_dir,
                            "bar_chart": bar_chart,
                            "radar_chart": radar_chart,
                        }
                        st.session_state.messages.append(err_msg)
        
                    st.toast("💾 已自动保存中断前的任务进度", icon="🛡️")
        
                # 真正中断后再解锁
                st.session_state.processing = False
                st.session_state.current_processing_query = ""
                st.rerun()
        
            else:
                st.session_state.processing = False
                st.session_state.current_processing_query = ""
                return
                
# =======================================================
# ⚙️ Task Processing Logic (带超时优化)
# =======================================================
def process_agent_task(usage_manager):
    """Core Agent Execution Logic with Global Timeout & Smart Retry"""

    # Ensure messages list exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        from Agent.Utils.file_utils import create_question_folder
        from Agent.Agent_Config.deepseek_client import update_client_settings
    except ImportError as e:
        st.error(f"❌ Core module import failed: {e}")
        st.session_state.processing = False
        return

    query = st.session_state.current_processing_query

    with st.chat_message("user"):
        st.markdown(query)

    # Append user msg only once
    if not st.session_state.messages or st.session_state.messages[-1].get("content") != query:
        st.session_state.messages.append({"role": "user", "content": query})
        usage_manager.log_question()

        # Requires upper layer to init session_run_dir / agent_engine
        session_run_dir = st.session_state.get("session_run_dir")
        if not session_run_dir:
            raise RuntimeError("session_run_dir is missing. Please reinitialize the session directory.")

        current_question_dir = create_question_folder(st.session_state.session_run_dir, query)
        st.session_state.last_run_dir = current_question_dir

    # Default placeholders
    bar_chart_path = None
    radar_chart_path = None
    extracted_scores = {}
    idea_visuals = []
    
    # Flags for loop
    is_task_completed = False
    
    with st.chat_message("assistant"):
        status = st.status("🧠 BEAVER is thinking... (Real-time Logs)", expanded=True)
    
        # 只创建一个固定日志槽位，后续反复覆盖这个槽位
        with status:
            log_slot = st.empty()
    
        original_stdout = sys.stdout
    
        current_dir = st.session_state.last_run_dir
        log_path = os.path.join(current_dir, "streamlit_live.log") if current_dir else None
        st.session_state.last_log_path = log_path
    
        try:
            sys.stdout = StreamToStatus(log_slot, original_stdout, log_path=log_path)

            # ✅ 如果发生 rerun/重建，先把已有日志尾部“补显示”到新的 status 容器
            if log_path and os.path.exists(log_path):
                tail = _tail_file(log_path, max_chars=24000)
                if tail.strip():
                    status.write("🧩 检测到历史日志，已从文件恢复显示（尾部片段）：")
                    log_slot.code(tail, language="text")

            status.write("🔍 Initializing & Searching Literature...")
            
            api_key = st.session_state.get("custom_api_key", "")
            base_url = st.session_state.get("custom_base_url", "")
            
            if not api_key or not base_url:
                raise RuntimeError("API Key / Base URL not configured. Please check the sidebar.")
            
            last_key = st.session_state.get("last_runtime_key")
            last_url = st.session_state.get("last_runtime_url")
            
            if api_key != last_key or base_url != last_url:
                update_client_settings(api_key, base_url)
                st.session_state["last_runtime_key"] = api_key
                st.session_state["last_runtime_url"] = base_url

            # ==========================================================
            # 🔥 [核心修改] 带限流的重试循环 (Limited Retry Loop)
            # ==========================================================
            response_text = ""
            retry_count = 0
            # 设置最大重试次数，防止死循环导致系统被杀
            MAX_RETRIES = 5 
            
            # 定义网络错误关键词
            network_error_keywords = [
                "timeout", "connection", "connect", "proxy", "handshake", 
                "remote end closed", "empty response", "429", "500", "502", "503", "504"
            ]

            while not is_task_completed:
                try:
                    # 如果之前有过重试，更新状态让用户知道网络好了，正在继续
                    if retry_count > 0:
                        status.update(label=f"🚀 网络恢复，正在进行第 {retry_count} 次尝试...", state="running")
                        print(f"\n[System] 重试连接中 ({retry_count}/{MAX_RETRIES})...")

                    # ✅ Run Agent
                    response_text = st.session_state.agent_engine.run_one_step(
                        query,
                        enable_quality_loop=st.session_state.toggle_loop,
                        use_short_term=st.session_state.toggle_short,
                        use_long_term=st.session_state.toggle_long,
                        save_path=current_dir
                    )
                    
                    # 如果执行到这里没有报错，说明成功了
                    is_task_completed = True
                    
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # 🔍 检查是否为网络/API类错误
                    is_network_error = any(kw in error_str for kw in network_error_keywords)
                    
                    if is_network_error:
                        retry_count += 1
                        
                        # 🔥 超过最大重试次数，主动抛出异常，而不是等系统杀死进程
                        if retry_count > MAX_RETRIES:
                            raise Exception(f"❌ 重试次数过多 ({MAX_RETRIES}次)。请检查网络连接或 DeepSeek 响应是否过慢。")
                        
                        wait_seconds = 5 # 等待秒数
                        
                        # 更新 UI 状态
                        status.update(label=f"⚠️ 网络连接不稳定/超时，将在 {wait_seconds}秒 后重试...", state="running")
                        
                        # 写入日志 (StreamToStatus 会自动显示在 UI 和 Log 文件中)
                        print(f"\n[Network Error] 检测到异常: {str(e)}")
                        print(f"[System] 正在等待网络恢复... (尝试 {retry_count}/{MAX_RETRIES})")
                        
                        # 强制刷新缓冲区，确保用户看到提示
                        sys.stdout.flush() 
                        
                        # 阻塞等待
                        time.sleep(wait_seconds)
                        continue # 跳过本次循环，重新尝试 run_one_step
                    else:
                        # ❌ 如果是代码逻辑错误 (KeyError, ValueError等)，直接抛出
                        raise e

            # ==========================================================
            # ⬇️ 以下逻辑保持不变 (任务成功后的处理)
            # ==========================================================
            
            status.write("✍️ Visualizing Data & Generating Report...")

            # Phase 1 & 2 charts
            bar_chart_path = _find_latest_file(current_dir, ["**/initial_screening_bar.png"])
            radar_chart_path = _find_latest_file(current_dir, ["**/final_verification_radar.png"])

            has_bar_chart = bool(bar_chart_path and os.path.exists(bar_chart_path))
            has_radar_chart = bool(radar_chart_path and os.path.exists(radar_chart_path))

            if has_bar_chart:
                status.write("📊 Generated Initial Screening Chart:")
                status.image(bar_chart_path, caption="Divergence Analysis (Intermediate Result)")

            if has_radar_chart:
                status.write("🧬 Generated Final Evaluation Chart:")
                status.image(radar_chart_path, caption="Convergence Analysis (Intermediate Result)")

            status.update(label="✅ Task Completed!", state="complete", expanded=False)

            # ---- Final display ----
            if has_bar_chart:
                st.markdown("### 📉 Phase 1: Divergence Analysis")
                st.image(bar_chart_path, caption="📉 Initial Screening: Divergence Analysis", width="stretch")

            st.markdown(response_text)

            # Load execution_context + idea visuals
            execution_context = _load_execution_context(current_dir)
            idea_visuals = execution_context.get("idea_visuals", []) or []
            _render_idea_visuals(idea_visuals, title="📊 Idea Visualizations")

            # Fallback for radar chart scores
            extracted_scores = {}
            if not has_radar_chart:
                try:
                    json_files = glob.glob(os.path.join(current_dir, "**", "*.json"), recursive=True)
                    for jf in json_files:
                        with open(jf, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, dict) and "scores_dict" in data:
                            extracted_scores = data["scores_dict"]
                            break
                except Exception:
                    extracted_scores = {}

            if has_radar_chart:
                st.markdown("### 🧬 Innovation Assessment")
                st.image(radar_chart_path, caption="🧬 Final Verification: Multi-Dimensional Evaluation", width="stretch")
            elif extracted_scores:
                st.markdown("### 🧬 Innovation Assessment")
                fig = render_four_dim_radar(extracted_scores)
                st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

            # Download Section
            if current_dir and os.path.exists(current_dir):
                st.divider()
                render_download_section(current_dir)

            # ✅ Save to history（把 log_path 也存进去，rerun 后能恢复）
            msg_payload = {
                "role": "assistant",
                "content": response_text,
                "dir_path": current_dir,
                "scores": extracted_scores,
                "bar_chart": bar_chart_path if has_bar_chart else None,
                "radar_chart": radar_chart_path if has_radar_chart else None,
                "ashby_chart": None,
                "pca_chart": None,
                "idea_visuals": idea_visuals,
                "log_path": log_path,  # ✅ NEW
            }
            st.session_state.messages.append(msg_payload)
        
        except Exception as e:
            status.update(label="❌ Error Occurred", state="error", expanded=True)
            st.error(f"Runtime Error: {e}")
            st.code(traceback.format_exc())

            # 把错误也写入历史（并带 log_path）
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 任务运行出错: {str(e)}",
                "dir_path": current_dir,
                "log_path": log_path,
            })

        finally:
            # ✅ 一定要恢复 stdout
            sys.stdout = original_stdout

            # ✅ 防止 death loop
            st.session_state.processing = False
            st.session_state.current_processing_query = ""
