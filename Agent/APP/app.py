import streamlit as st
import os
import traceback # 用于打印详细报错
import base64    # 🔥 新增：用于处理图片编码

# ==========================================
# 🛡️ 路径修复核心 (自动定位当前脚本所在文件夹)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "BEAVER-LOGO.png")

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="BEAVER Platform",
    page_icon=logo_path,
    layout="wide"
)

# === 导入认证模块 ===
import modules.auth as auth

# 初始化 Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ==========================================
# 🛠️ 辅助函数：将本地图片转换为 Base64 编码
# ==========================================
def get_img_as_base64(file_path):
    """读取图片文件并转换为 base64 字符串，以便在 HTML 中使用"""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# === 核心逻辑分流 ===
if not st.session_state['logged_in']:
    auth.init_db()
    auth.render_login_interface()

else:
    # 🛡️ 给主程序加个保险，防止白屏
    try:
        # 2. 导入模块
        from modules.setup import configure_paths, auto_configure_remote_access
        from modules.utils import UsageManager
        from modules.sidebar import render_sidebar
        from modules.engine import init_agent_engine, init_session_dir
        from modules.chat import render_chat_interface

        # 3. 系统级初始化
        configure_paths()              
        auto_configure_remote_access() 

        # 4. 加载 CSS (保留原有样式，新增部分在下方 HTML 中直接内联)
        st.markdown("""
            <style>
                html, body, [class*="css"] { font-family: 'Times New Roman', Times, serif; }
                h1, h2, h3, h4, h5, h6 { font-family: 'Times New Roman', Times, serif; }
                .stMarkdown, .stText, p { font-family: 'Times New Roman', Times, serif !important; }
                div[data-testid="stMetricValue"] { font-size: 24px; color: #0068c9; }
                div[data-testid="stDownloadButton"] button {
                    width: 100%;
                    border: 1px solid #e0e0e0;
                    background-color: #f9f9f9;
                    color: #333;
                    transition: all 0.3s;
                }
                div[data-testid="stDownloadButton"] button:hover {
                    border-color: #0068c9;
                    color: #0068c9;
                    background-color: #eef6ff;
                }
            .stChatMessage { max-width: 100% !important; }
            </style>
            """, unsafe_allow_html=True)

        # ==========================================
        # 5. 应用逻辑 (🔥 使用 Flexbox 布局修复重叠问题)
        # ==========================================
        
        # 1. 转换图片
        if os.path.exists(logo_path):
            img_b64 = get_img_as_base64(logo_path)
            # HTML 模板：使用 Flexbox 实现左图右文，gap 控制间距，align-items 实现垂直居中
            header_html = f"""
            <div style="display: flex; align-items: center; gap: 25px; margin-bottom: 20px;">
                <img src="data:image/png;base64,{img_b64}" style="width: 110px; height: auto; border-radius: 50%;">
                <h1 style="margin: 0; padding: 0; font-size: 3.0rem; line-height: 1.2;">BEAVER Platform</h1>
            </div>
            """
            st.markdown(header_html, unsafe_allow_html=True)
        else:
            # 降级处理：如果找不到图，只显示标题
            st.error(f"Logo not found: {logo_path}")
            st.title("BEAVER Platform")

        # 欢迎语和副标题
        user_name = st.session_state.get('username', 'User')
        st.caption(f"Welcome, {user_name} | Powered by Multi-Agent System & Local Vector Database")

        # 初始化统计
        usage_manager = UsageManager()
        usage_manager.log_visit()

        # 渲染侧边栏
        render_sidebar(usage_manager)
        
        with st.sidebar:
            st.divider()
            # 管理员专属区域
            if st.session_state.get('role') == 'admin':
                with st.expander("🕵️ 管理员后台 (Admin Only)"):
                    st.write("### 注册用户概览")
                    try:
                        df_users = auth.get_all_users()
                        st.dataframe(df_users, hide_index=True)
                        st.write(f"总用户数: {len(df_users)}")
                    except Exception as e:
                        st.error(f"数据库读取失败: {e}")
            
            # 退出按钮
            if st.button("🚪 退出登录", use_container_width=True):
                st.session_state['logged_in'] = False
                st.rerun()

        # 初始化引擎与目录
        init_session_dir(current_dir)
        init_agent_engine(current_dir)

        # 渲染主聊天界面
        render_chat_interface(usage_manager)

    except Exception as e:
        # 🚨 捕获所有让屏幕变白的错误
        st.error("💥 系统发生严重错误 (System Crash)")
        st.error(f"错误信息: {str(e)}")
        with st.expander("查看详细报错 (Traceback)", expanded=True):
            st.code(traceback.format_exc())
            if st.button("尝试刷新页面"):
                st.rerun()