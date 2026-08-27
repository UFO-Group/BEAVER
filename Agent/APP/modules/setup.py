import os
import sys

def configure_paths():
    """配置系统路径，确保能导入 Agent 和项目根目录模块"""
    # 1. 获取 modules 文件夹的绝对路径
    current_file_path = os.path.abspath(__file__)
    modules_dir = os.path.dirname(current_file_path)
    
    # 2. 获取 app.py 所在的目录 (modules 的上一级)
    app_dir = os.path.dirname(modules_dir)
    
    # 3. 获取 Agent 包的目录 (app.py 的上一级)
    agent_pkg_dir = os.path.dirname(app_dir)
    
    # 4. 获取 项目根目录 (Agent 的上一级)
    project_root = os.path.dirname(agent_pkg_dir)

    # 打印调试信息 (如果你还遇到报错，请看控制台打印出的路径对不对)
    # print(f"[Debug] Path Patching:\nRoot: {project_root}\nAgent: {agent_pkg_dir}")

    paths_to_add = [project_root, agent_pkg_dir, app_dir]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)

def auto_configure_remote_access():
    """自动配置 .streamlit/config.toml 以允许远程访问"""
    # 获取 app.py 所在目录 (即当前工作目录，因为我们是在 app.py 运行的)
    current_dir = os.getcwd() 
    config_dir = os.path.join(current_dir, ".streamlit")
    config_path = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_path):
        try:
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            config_content = """
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false
runOnSave = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[runner]
magicEnabled = false
"""
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(config_content.strip())
            print("✅ [Auto-Config] 已自动配置远程访问权限")
        except Exception as e:
            print(f"❌ 无法自动创建配置文件: {e}")