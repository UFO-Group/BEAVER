# Agent/Utils/text_utils.py

import re
import json
import ast
import logging
from rich.console import Console

# Configure logger
logger = logging.getLogger(__name__)
console = Console()

def clean_llm_json_string(text: str) -> str:
    """
    🧹 清洗 LLM 输出，专门适配 DeepSeek-R1 的 <think> 标签，防止 JSON 解析崩溃
    """
    if not text:
        return ""

    # 1. 🔥 [核心修复] 强力移除 <think>...</think> 思考过程
    # flags=re.DOTALL 让 . 能匹配换行符，务必加上
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    text = text.strip()

    # 2. 尝试匹配 Markdown 代码块 ```json ... ```
    markdown_pattern = r"```(?:json|JSON)?\s*(.*?)\s*```"
    match = re.search(markdown_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # 3. 如果没有 Markdown，暴力寻找最外层的 {} 或 []
    start_brace = text.find("{")
    start_bracket = text.find("[")
    
    start_idx = -1
    end_char = ""
    
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
        end_char = "}"
    elif start_bracket != -1:
        start_idx = start_bracket
        end_char = "]"
        
    if start_idx != -1:
        end_idx = text.rfind(end_char)
        if end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1].strip()

    return text.strip()

def parse_json_safely(text: str):
    """
    🛡️ Robustly parse JSON from LLM output.
    Returns the parsed object (dict/list) or None if failed.
    
    Strategy:
    1. Clean the string (extract from markdown/brackets).
    2. Try standard json.loads (strict).
    3. Try ast.literal_eval (lenient, handles single quotes).
    4. Try Regex Repair (handles trailing commas).
    """
    if not text:
        return None

    # Step 1: Basic Extraction
    cleaned_text = clean_llm_json_string(text)
    if not cleaned_text:
        return None

    # Strategy A: Standard JSON
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # Strategy B: Python AST (handles {'key': 'val'} which JSON hates)
    try:
        return ast.literal_eval(cleaned_text)
    except (ValueError, SyntaxError):
        pass
        
    # Strategy C: JSON Repair (Fix common LLM syntax errors)
    # Fix 1: Trailing commas (e.g., {"a": 1,} -> {"a": 1})
    try:
        repaired_text = re.sub(r',\s*\}', '}', cleaned_text)
        repaired_text = re.sub(r',\s*\]', ']', repaired_text)
        return json.loads(repaired_text)
    except json.JSONDecodeError:
        pass

    # Strategy D: Fix unquoted keys (simple cases only)
    # e.g. {key: "value"} -> {"key": "value"}
    try:
        # This is a bit risky but works for simple keys
        repaired_text = re.sub(r'(\s*[{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', cleaned_text)
        return json.loads(repaired_text)
    except:
        pass

    # If all fail:
    # console.print(f"[dim]⚠️ JSON Parsing Failed. Content start: {cleaned_text[:50]}...[/dim]")
    return None

def extract_citations(text: str) -> list:
    """
    🔍 Extract citation numbers or IDs.
    Supports: [1], [1, 2], [Author, 2024]
    """
    if not text:
        return []
    
    # Match content inside square brackets
    pattern = r"\[(.*?)\]"
    citations = re.findall(pattern, text)
    
    # Filter out common false positives (like image tags or markdown links)
    clean_citations = []
    for c in citations:
        c_lower = c.lower()
        if c_lower.startswith("image of") or c_lower == "x" or "fig." in c_lower:
            continue
        clean_citations.append(c)
        
    return clean_citations

def extract_code_block(text: str, language: str = "python") -> str:
    """
    💻 Extract code block for a specific language.
    """
    pattern = rf"```{language}(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def remove_markdown_bold(text: str) -> str:
    """
    📝 Remove Markdown bolding (**text** -> text).
    Useful for cleaning up titles for filenames or Word docs.
    """
    if not text: return ""
    return text.replace("**", "").replace("__", "").replace("##", "")