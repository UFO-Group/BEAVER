import os
import time
import streamlit as st
from datetime import datetime

# =========================================================
# 🔥 [核心修复]: 全局内存缓存
# 无论多少用户、怎么刷新，这个函数在整个服务器生命周期内只执行一次！
# =========================================================
@st.cache_resource(show_spinner=False)
def warmup_retrieval_backend():
    try:
        from Agent.RAG.vector_store import load_vector_store
        # 真正的“读硬盘缓存”只在这里发生一次，读完就永远驻留在 RAM 里
        load_vector_store()
        return True
    except Exception as e:
        print(f"⚠️ Vector store warmup failed: {e}")
        return False

def init_agent_engine(project_root):
    """初始化 ResearchAgent"""
    
    # 1. 检查 API 是否就绪
    if not st.session_state.get("api_ready", False):
        st.warning("👈 **系统锁定**：请在左侧侧边栏输入 **Base URL** 和 **API Key** 以登录系统。")
        st.info("提示：输入后系统将自动连接。")
        st.stop()

    # 2. 初始化引擎
    if "agent_engine" not in st.session_state:
        # 延迟导入，防止路径未配置好时报错
        try:
            from Agent.Planner.planner_agent import ResearchAgent
            from Agent.Agent_Config.deepseek_client import update_client_settings
        except ImportError:
            st.error("❌ 无法导入 ResearchAgent，请检查路径配置。")
            st.stop()
            
        st.info("⏳ 正在初始化系统检索组件。由于需要加载语料索引并启动 BM25 检索器，首次启动时间可能较长。")

        with st.spinner("🚀 API Key 验证通过，正在启动科研智能体引擎..."):
            try:
                # 再次确认配置
                update_client_settings(st.session_state["custom_api_key"], st.session_state["custom_base_url"])

                # 🔥 [核心修复]: 在创建 Agent 前，先调用预热函数
                # 如果是第一次，它会卡一下读硬盘；如果是刷新页面，这里瞬间就过去了
                warmup_retrieval_backend()

                st.session_state.agent_engine = ResearchAgent(
                    use_memory=True,
                    enable_short_term=True,
                    enable_long_term=True
                )
                st.success("Agent Engine Ready!")
                time.sleep(0.5)
                st.rerun() 
            except Exception as e:
                st.error(f"❌ 核心模块加载失败: {e}")
                st.stop()

def init_session_dir(project_root):
    """初始化会话运行目录"""
    if "session_run_dir" not in st.session_state:
        # 使用 Agent 目录下的统一会话输出目录
        target_base_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Session_Runs")
        )
        if not os.path.exists(target_base_root):
            try:
                os.makedirs(target_base_root, exist_ok=True)
            except Exception:
                target_base_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "Session_Runs")
                )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"Run_{timestamp}"
        session_run_path = os.path.join(target_base_root, run_dir_name)
        os.makedirs(session_run_path, exist_ok=True)
        st.session_state.session_run_dir = session_run_path
