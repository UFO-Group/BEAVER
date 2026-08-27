import streamlit as st
import sys

def render_sidebar(usage_manager):
    """渲染侧边栏"""
    
    with st.sidebar:
        st.title("🎛️ Control Panel")
        disabled = st.session_state.get("processing", False)
        
        # --- API 配置 (已修改防弹窗逻辑) ---
        render_api_config(disabled=disabled)
        
        # --- Agent 配置 ---
        st.markdown("### ⚙️ Agent Configuration")
        with st.container(border=True):
            render_memory_toggles()
            st.divider()
            render_reasoning_toggles()

        # --- 重置按钮 ---
        st.write("")
        if st.button(
            "🗑️ Reset Conversation",
            type="primary",
            use_container_width=True,
            disabled=disabled
        ):
            if "agent_engine" in st.session_state:
                try:
                    st.session_state.agent_engine.run_one_step("/clear")
                except:
                    pass
        
            st.session_state.messages = []
            st.session_state["last_run_dir"] = None
            st.session_state["processing"] = False
            st.session_state["current_processing_query"] = ""
            st.session_state["last_log_path"] = None
            st.session_state["run_token"] = False
            st.rerun()

        # --- 统计数据 ---
        st.markdown("---")
        latest_stats = usage_manager.get_stats()
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; font-size: 11px; color: #666; margin-top: 10px; border: 1px solid #eee;">
                <div style="font-weight: 600; margin-bottom: 4px; color: #444;">📊 Server Stats</div>
                <div style="display: flex; justify-content: space-between;">
                    <span>👥 Visits: <b>{latest_stats['total_visits']}</b></span>
                    <span>💬 Tasks: <b>{latest_stats['total_questions']}</b></span>
                </div>
                <div style="font-size: 9px; color: #999; margin-top: 4px; text-align: right;">
                    Last: {latest_stats['last_active_time']}
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_api_config(disabled=False):
    # 延迟导入：防止循环依赖或路径错误
    try:
        from Agent.Agent_Config.deepseek_client import update_client_settings
    except ImportError:
        st.error("❌ 无法导入 update_client_settings。请检查 modules/setup.py 中的路径配置是否正确。")
        st.code(f"Current sys.path: {sys.path}", language="text") 
        return

    with st.expander("🔑 API Configuration (Required)", expanded=True):
        st.caption("Please input your API details to login")
        st.caption("🔒 **Security Note**: stored locally in session memory.")
        
        # 🔥🔥🔥 [核心修改 1] 使用 st.form 包裹输入框
        # 这样可以防止输入过程中失去焦点导致的页面自动刷新（即防止“信息抹掉”）
        with st.form(key="api_config_form"):
            
            user_base_url = st.text_input(
                "Base URL", 
                value=st.session_state.get("custom_base_url", ""), 
                placeholder="your_URL",
                autocomplete="off",
                disabled=disabled
            )
            
            # 插入一个空行，在 DOM 结构上打断“用户名+密码”的连续性，减少浏览器误判
            st.write("") 

            # 🔥🔥🔥 [核心修改 2] 属性欺骗
            user_api_key = st.text_input(
                label="Security Token",       
                value=st.session_state.get("custom_api_key", ""), 
                type="password", 
                key="safe_access_token_v2",   
                placeholder="sk-...",
                help="输入访问令牌",
                # 关键：将 autocomplete 设为 "one-time-code"
                # 这告诉 Edge：“这是2FA验证码，不是密码，别问我要不要保存”
                autocomplete="one-time-code",
                disabled=disabled   
            )
            
            # 提交按钮
            submitted = st.form_submit_button("🔌 连接系统 (Connect)", disabled=disabled)

        # --- 逻辑处理移到 form 外部，只有点击按钮后才执行 ---
        if submitted:
            # 更新 Session State
            st.session_state["custom_api_key"] = user_api_key
            st.session_state["custom_base_url"] = user_base_url

            if user_api_key and user_base_url:
                is_success = update_client_settings(user_api_key, user_base_url)
                if is_success:
                    st.session_state["api_ready"] = True
                    st.session_state["last_configured_key"] = user_api_key
                    st.session_state["last_configured_url"] = user_base_url
                
                    st.session_state.pop("last_runtime_key", None)
                    st.session_state.pop("last_runtime_url", None)
                
                    st.success("✅ Connected!")
                    st.rerun()
                else:
                    st.session_state["api_ready"] = False
                    st.error("❌ Invalid Config (连接失败)")
            else:
                st.warning("⚠️ 请输入完整的 URL 和 Key")

        # 显示连接状态（非提交时的回显）
        elif st.session_state.get("api_ready"):
             st.success("✅ Connected (Cached)")
            
def render_memory_toggles():
    st.caption("🧠 Memory Architecture")

    def on_memory_change():
        if st.session_state.toggle_short:
            st.toast("✅ Short-Term Memory ENABLED", icon="🧠")
            sys.__stdout__.write("✅ [Config] Short-Term Memory: ON\n")
        else:
            st.toast("🚫 Short-Term Memory DISABLED", icon="💤")
            sys.__stdout__.write("🚫 [Config] Short-Term Memory: OFF\n")

    def on_long_change():
        if st.session_state.toggle_long:
            st.toast("✅ Long-Term Memory ENABLED", icon="📚")
            sys.__stdout__.write("✅ [Config] Long-Term Memory: ON\n")
        else:
            st.toast("🚫 Long-Term Memory DISABLED", icon="🗑️")
            sys.__stdout__.write("🚫 [Config] Long-Term Memory: OFF\n")

    st.toggle(
        "Short-Term Memory",
        value=False,
        key="toggle_short",
        on_change=on_memory_change,
        disabled=st.session_state.get("processing", False)
    )
    st.toggle(
        "Long-Term Memory",
        value=False,
        key="toggle_long",
        on_change=on_long_change,
        disabled=st.session_state.get("processing", False)
    )

def render_reasoning_toggles():
    st.caption("🤔 Reasoning Strategy")

    def on_loop_change():
        if st.session_state.toggle_loop:
            st.toast("✅ Deep Reflection Mode ON", icon="🔄")
            sys.__stdout__.write("✅ [Config] Deep Reflection Mode: ON\n")
        else:
            st.toast("🚫 Deep Reflection Mode OFF", icon="⚡")
            sys.__stdout__.write("🚫 [Config] Deep Reflection Mode: OFF\n")

    st.toggle(
        "Deep Reflection Mode",
        value=False,
        key="toggle_loop",
        on_change=on_loop_change,
        disabled=st.session_state.get("processing", False)
    )