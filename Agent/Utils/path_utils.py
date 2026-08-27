# Agent/Utils/path_utils.py
import re

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