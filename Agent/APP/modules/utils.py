import os
import json
import re
import sys
import time
import logging
import threading
import streamlit as st
from datetime import datetime

# ==========================================
# 🔇 [核心修复] 强力压制线程上下文警告
# ==========================================
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

# 尝试导入上下文获取函数
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    try:
        from streamlit.scriptrunner import get_script_run_ctx
    except ImportError:
        get_script_run_ctx = None

# ==========================================
# 🔥 [工具] 路径清洗 (供全局调用)
# ==========================================
def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    清洗文件名，防止 Windows 路径报错 (非法字符/尾部空格)
    """
    if not name:
        return "Untitled_Task"

    # 1. 替换非法字符为下划线
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", name)
    # 2. 替换连续空格/换行符为单个下划线
    safe_name = re.sub(r"\s+", "_", safe_name)
    # 3. 去除首尾空白和下划线
    safe_name = safe_name.strip().strip("_")
    return safe_name[:max_length]


# ==========================================
# 📊 统计管理器
# ==========================================
class UsageManager:
    def __init__(self, log_file="app_usage_stats.json"):
        self.log_path = os.path.abspath(log_file)
        self._init_log_file()

    def _init_log_file(self):
        if not os.path.exists(self.log_path):
            initial_data = {
                "total_visits": 0,
                "total_questions": 0,
                "last_active_time": "",
                "visitor_ips": []
            }
            try:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump(initial_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                sys.__stdout__.write(f"Error initializing log file: {e}\n")

    def _load_data(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"total_visits": 0, "total_questions": 0, "last_active_time": ""}

    def _save_data(self, data):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            sys.__stdout__.write(f"Error saving stats: {e}\n")

    def log_visit(self):
        if "has_counted_visit" not in st.session_state:
            data = self._load_data()
            data["total_visits"] += 1
            data["last_active_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_data(data)
            st.session_state.has_counted_visit = True
            return data
        return self._load_data()

    def log_question(self):
        data = self._load_data()
        data["total_questions"] += 1
        data["last_active_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_data(data)
        return data

    def get_stats(self):
        return self._load_data()


# ==========================================
# 📝 日志重定向工具（UI + 落盘）
# ==========================================
class StreamToStatus:
    def __init__(
        self,
        log_placeholder,
        original_stdout,
        log_path: str = None,
        max_bytes: int = 2_000_000,
        max_lines: int = 300,
    ):
        self.log_placeholder = log_placeholder
        self.original = original_stdout if original_stdout is not None else sys.__stdout__
        self.log_path = log_path
        self.max_bytes = max_bytes
        self.max_lines = max_lines
        self.lines = []
        
        # 2️⃣ [新增] UI 渲染节流控制：记录上次刷新时间和时间间隔
        self.last_render_time = time.time()
        self.render_interval = 0.5  # 每 0.5 秒最多向前端发一次更新指令

        self.main_thread_id = threading.get_ident()

        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self._skip_patterns = ("HTTP Request",)

        if self.log_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            except Exception:
                pass

    def _append_log(self, text: str):
        if not self.log_path:
            return

        clean_text_for_file = self.ansi_escape.sub("", str(text))

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(clean_text_for_file)

            if self.max_bytes and self.max_bytes > 0:
                try:
                    if os.path.getsize(self.log_path) > self.max_bytes:
                        with open(self.log_path, "rb") as rf:
                            rf.seek(-min(self.max_bytes // 2, os.path.getsize(self.log_path)), os.SEEK_END)
                            tail = rf.read()
                        with open(self.log_path, "wb") as wf:
                            wf.write(tail)
                except Exception:
                    pass
        except Exception:
            try:
                sys.__stdout__.write("[StreamToStatus] Failed to append log.\n")
            except Exception:
                pass

    def _should_show(self, clean_text: str) -> bool:
        allowed_prefixes = (
            "🧩", "🚀", "🧭", "🧠", "🔧", "➡", "🔍", "📝", "📊", "🧊", "📉", "🧮",
            "🧬", "✅", "❌", "✔", "⚠️", "🏆", "🧪", "💡", "💾", "🧹",
            "Step", "Pipeline"
        )
        return clean_text.startswith(allowed_prefixes)

    def _render(self):
        try:
            current_time = time.time()
            # 3️⃣ [新增逻辑] 只有距离上次刷新超过 0.5 秒，才真正向网页前端发送更新
            if current_time - self.last_render_time > self.render_interval:
                text = "\n".join(self.lines[-self.max_lines:])
                self.log_placeholder.code(text, language="text")
                self.last_render_time = current_time
        except Exception:
            pass

    def write(self, text):
        if text is None:
            return 0

        text_str = str(text)

        # 先写终端
        try:
            self.original.write(text_str)
            self.original.flush()
        except Exception:
            pass

        # 再落盘
        self._append_log(text_str)

        # 👇 [新增的核心防御] 如果当前不是 Streamlit 主线程，直接 return，坚决不碰 UI
        if threading.get_ident() != self.main_thread_id:
            return len(text_str)
            
        # 子线程不碰 UI
        if get_script_run_ctx and get_script_run_ctx() is None:
            return len(text_str)

        clean_text = self.ansi_escape.sub("", text_str).strip()
        if not clean_text:
            return len(text_str)

        # 跳过无关日志
        if any(pat in clean_text for pat in self._skip_patterns):
            return len(text_str)

        if self._should_show(clean_text):
            self.lines.append(clean_text)
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]
            self._render()

        return len(text_str)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass