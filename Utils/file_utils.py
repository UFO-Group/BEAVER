# Agent/Utils/file_utils.py

import os
import sys
import json
import re  # <--- 必须导入，用于清洗文件名
from typing import Any, Optional
from rich.console import Console

# 尝试导入 Pandas，如果环境没装也不影响其他功能
try:
    import pandas as pd
except ImportError:
    pd = None

# 1. 尝试导入配置
try:
    import Agent.Agent_Config.agent_config as config
except ImportError:
    config = None

console = Console()

# =======================================================
#  Part 0: 路径与命名辅助函数 (关键新增部分)
# =======================================================

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    清洗字符串，使其可以作为合法的文件名/文件夹名。
    1. 去掉非法字符 (\ / : * ? " < > |)
    2. 去掉非 ASCII 字符 (如 °C 中的 °)，防止 Windows 编码问题
    3. 限制长度
    """
    if not text:
        return "Untitled"
    
    # 1. 替换 Windows/Linux 文件名中的非法字符为下划线
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", text)
    
    # 2. 🔥 关键：替换掉任何非 ASCII 字符（比如 ° 符号），防止编码导致路径找不到
    # 这一行会把 "200 °C" 变成 "200 _C"
    sanitized = re.sub(r'[^\x00-\x7F]+', '_', sanitized)
    
    # 3. 去掉换行符，将连续空格/下划线替换为单个下划线
    sanitized = sanitized.replace('\n', '_').replace('\r', '')
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    
    # 4. 去除首尾空白和下划线
    sanitized = sanitized.strip("_")
    
    # 5. 限制长度，防止文件名过长报错
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def create_question_folder(base_run_dir: str, question_text: str) -> str:
    """
    在 base_run_dir 下创建一个以 question_text 命名的文件夹。
    如果重名，自动添加 -1, -2, -3 后缀。
    返回: 创建好的完整文件夹路径
    """
    # 1. 清洗问题文本作为文件夹名
    folder_name = sanitize_filename(question_text, max_length=50)
    
    # 2. 初始目标路径
    target_path = os.path.join(base_run_dir, folder_name)
    
    # 3. 重名检测与递增处理
    if os.path.exists(target_path):
        counter = 1
        while True:
            # 尝试生成新名字：FolderName-1, FolderName-2
            new_folder_name = f"{folder_name}-{counter}"
            new_target_path = os.path.join(base_run_dir, new_folder_name)
            
            if not os.path.exists(new_target_path):
                target_path = new_target_path
                break
            counter += 1
            
    # 4. 创建文件夹
    try:
        os.makedirs(target_path, exist_ok=True)
    except Exception as e:
        console.print(f"[red]❌ 文件夹创建失败: {e}[/red]")
        # 兜底：如果创建失败，回退到 base_run_dir
        return base_run_dir
        
    return target_path

# =======================================================
#  Part 1: 结果保存函数 (修复路径不存在的问题)
# =======================================================

def save_step_result(
    step_id: str,
    step_type: str,
    content: Any,
    step_log_dir: Optional[str] = None, 
    console: Optional[Console] = None,
    suffix: str = ""
) -> None:
    """
    保存步骤结果到指定目录。
    核心逻辑：如果你传入了 step_log_dir，我就存到那里；如果没传，我就用 Config 里的默认路径。
    """

    # -------------------------------------------------------
    # 1. 智能路径获取 (支持分文件夹策略的核心)
    # -------------------------------------------------------
    if step_log_dir is None:
        # 如果调用方没传路径，就尝试读取配置文件的默认路径
        if config and hasattr(config, "STEP_LOG_DIR"):
            step_log_dir = config.STEP_LOG_DIR
        else:
            step_log_dir = "step_logs_fallback"

    # 规范化路径
    step_log_dir = os.path.normpath(step_log_dir)
    
    # 🔥🔥🔥 核心修复：每次写入前必须强制确保目录存在！🔥🔥🔥
    if not os.path.exists(step_log_dir):
        try:
            os.makedirs(step_log_dir, exist_ok=True)
        except Exception as e:
            if console:
                console.print(f"[red]❌ 无法创建目录 {step_log_dir}: {e}[/red]")
            raise RuntimeError(f"无法创建目录 {step_log_dir}: {e}") from e

    # 安全处理文件名 (使用 sanitize_filename 而不是简单的 replace)
    safe_id = sanitize_filename(str(step_id), max_length=30)
    safe_type = sanitize_filename(str(step_type), max_length=30)
    
    # -------------------------------------------------------
    # 2. 后缀处理逻辑
    # -------------------------------------------------------
    file_ext = ".txt" # 默认文本后缀
    file_suffix = ""

    # 如果 suffix 是 "md"，则改变扩展名
    if suffix and suffix.lower() == "md":
        file_ext = ".md"
        file_suffix = "" 
    elif suffix:
        # 清洗 suffix，防止含有非法字符
        clean_suffix = sanitize_filename(suffix, max_length=10)
        file_suffix = f"_{clean_suffix}"

    # -------------------------------------------------------
    # 3. 分类保存逻辑
    # -------------------------------------------------------

    # === A) 表格类：list[dict] -> 保存为 CSV 和 JSON ===
    if isinstance(content, list) and content and isinstance(content[0], dict):
        if pd:
            df = pd.DataFrame(content)

            csv_name = f"{safe_id}_{safe_type}{file_suffix}.csv"
            csv_path = os.path.join(step_log_dir, csv_name)

            json_name = f"{safe_id}_{safe_type}{file_suffix}.json"
            json_path = os.path.join(step_log_dir, json_name)

            try:
                print(f"[DEBUG save_step_result] writing csv -> {csv_path}", flush=True)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"[DEBUG save_step_result] saved csv -> {csv_name}", flush=True)
            
                print(f"[DEBUG save_step_result] writing json -> {json_path}", flush=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
                    f.flush()
                    # os.fsync(f.fileno())  # 临时关闭
                print(f"[DEBUG save_step_result] saved json -> {json_name}", flush=True)
            
                if console:
                    console.print(f"[dim]💾 表格已保存: {os.path.basename(csv_path)}[/dim]")
            
            except Exception as e:
                if console:
                    console.print(f"[red]❌ 保存表格失败: {e}[/red]")
                print(f"[DEBUG save_step_result] table save failed: {repr(e)}", flush=True)
                raise RuntimeError(f"保存表格失败: {csv_path}") from e
                
        else:
            if console:
                console.print("[yellow]⚠️ 未安装 Pandas，仅保存为 JSON[/yellow]")

            json_path = os.path.join(step_log_dir, f"{safe_id}_{safe_type}{file_suffix}.json")
            try:
                print(f"[DEBUG save_step_result] writing json only -> {json_path}", flush=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
                    f.flush()
                    # os.fsync(f.fileno())
            except Exception as e:
                print(f"[DEBUG save_step_result] json-only save failed: {repr(e)}", flush=True)
                raise RuntimeError(f"保存 JSON 失败: {json_path}") from e
        return

    # === B) 文本类：str -> 保存为 TXT 或 MD ===
    if isinstance(content, str):
        file_name = f"{safe_id}_{safe_type}{file_suffix}{file_ext}"
        txt_path = os.path.join(step_log_dir, file_name)
        try:
            print(f"[DEBUG save_step_result] writing text -> {txt_path}", flush=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                # os.fsync(f.fileno())

            print(f"[DEBUG save_step_result] saved text -> {file_name}", flush=True)
            if console:
                console.print(f"[dim]💾 文本已保存: {os.path.basename(txt_path)}[/dim]")
        except Exception as e:
            if console:
                console.print(f"[red]❌ 保存文本失败: {e}[/red]")
            print(f"[DEBUG save_step_result] text save failed: {repr(e)}", flush=True)
            raise RuntimeError(f"保存文本失败: {txt_path}") from e
        return

    # === C) 其他类型（Dict等） -> 保存为 JSON ===
    json_path = os.path.join(step_log_dir, f"{safe_id}_{safe_type}{file_suffix}.json")
    try:
        print(f"[DEBUG save_step_result] writing generic json -> {json_path}", flush=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
            f.flush()
            # os.fsync(f.fileno())
    
        print(f"[DEBUG save_step_result] saved generic json -> {os.path.basename(json_path)}", flush=True)
        if console:
            console.print(f"[dim]💾 通用数据已保存: {os.path.basename(json_path)}[/dim]")
    
    except Exception as e:
        print(f"[DEBUG save_step_result] generic json save failed: {repr(e)}", flush=True)
        # 如果连 JSON 都转不了，强行转字符串存起来，防止数据丢失
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(str(content))
                f.flush()
                # os.fsync(f.fileno())
            print(f"[DEBUG save_step_result] fallback text saved -> {os.path.basename(json_path)}", flush=True)
            if console:
                console.print(f"[yellow]⚠️ 非标准对象，已存为文本: {os.path.basename(json_path)}[/yellow]")
        except Exception as e2:
            print(f"[DEBUG save_step_result] fallback save failed: {repr(e2)}", flush=True)
            if console:
                console.print(f"[red]❌ 彻底保存失败: {e2}[/red]")
            raise RuntimeError(f"彻底保存失败: {json_path}") from e2

# 🔥🔥🔥 [新增] 通用日志屏蔽工具 🔥🔥🔥
class NoStreamlitLog:
    """
    上下文管理器：强制将 Rich Console 和 print 重定向回后台终端。
    用于防止 Streamlit 在多线程并发时崩溃。
    """
    def __init__(self, agent_instance=None):
        self.agent = agent_instance
        self.original_stdout = sys.stdout
        self.original_console_file = None
        self.original_agent_console_file = None

    def __enter__(self):
        # 1. 尝试导入全局 console 对象
        try:
            from Agent.Agent_Config.agent_config import console as global_console
            self.original_console_file = global_console.file
            global_console.file = sys.__stdout__ # 强行指向黑窗口
        except ImportError:
            pass

        # 2. 处理 Agent 自己的 console
        if self.agent and hasattr(self.agent, 'console'):
            self.original_agent_console_file = self.agent.console.file
            self.agent.console.file = sys.__stdout__

        # 3. 还原标准 print
        if hasattr(sys.stdout, "original"):
            sys.stdout = sys.stdout.original
        else:
            sys.stdout = sys.__stdout__

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. 恢复全局 console
        try:
            from Agent.Agent_Config.agent_config import console as global_console
            if self.original_console_file:
                global_console.file = self.original_console_file
        except ImportError:
            pass
        
        # 2. 恢复 agent console
        if self.agent and self.original_agent_console_file:
            self.agent.console.file = self.original_agent_console_file

        # 3. 恢复标准 print
        sys.stdout = self.original_stdout