# Agent/Agent_Config/deepseek_client.py
import re
import os
import time
import random
import numpy as np
from openai import OpenAI

# 1. 确保所有模型变量都从配置中导入
from .agent_config import (
    EMBED_MODEL, 
    LLM_MODEL, 
    INTENT_MODEL, 
    DESIGN_MODEL, 
    SCORE_MODEL, 
    PLANNER_Module_MODEL,
    REPORT_MODEL
)

# =========================================================================
# 🔑 API 客户端配置
# =========================================================================

client = None

def update_client_settings(api_key: str, base_url: str):
    """
    允许外部 (Streamlit) 动态初始化 client
    """
    global client
    if not api_key or not base_url:
        print("⚠️ API Key 或 Base URL 为空，无法初始化 Client。")
        return False

    print(f"🔄 [DeepSeek] 正在配置 Client...")
    print("   -> API endpoint: configured")
    print("   -> API Key: configured")
    
    try:
        # 🔥🔥🔥 [核心修复] 设置超长超时时间 (600秒/10分钟)
        # DeepSeek R1 写长文 + 思考非常耗时，默认 60s 必挂
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=600.0,  # <--- 这里是关键！
            max_retries=2   # 库内部重试
        )
        print("✅ Client 配置成功！")
        return True
    except Exception as e:
        print(f"❌ Client 配置失败: {e}")
        return False

# =========================================================================
# 🛠️ 核心工具：自动重试控制器
# =========================================================================

def _check_client():
    """检查 client 是否已初始化"""
    if client is None:
        print("❌ 错误：API Client 未初始化！请先在网页侧边栏输入 API Key。")
        return False
    return True

def _safe_get_content(response, func_name="Unknown"):
    """
    统一处理 API 返回内容，强制清洗思维链，防止 Streamlit 渲染崩溃
    """
    try:
        if not response:
            return ""

        message = response.choices[0].message
        content = message.content
        
        # 1. 尝试清洗 <think> 标签 (DeepSeek R1 特有)
        # 🔥🔥🔥 [核心修复] 必须彻底移除 <think> 标签，否则 Streamlit 渲染会崩 (ElementNode Error)
        if content:
            cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            if cleaned_content:
                return cleaned_content
            
            # 如果清洗后为空 (说明只有思考过程)，绝对不能返回带 <think> 的原始内容
            print(f"⚠️ [{func_name}] Warning: Model output contained ONLY reasoning. Suppressing to prevent UI crash.")
            return ""

        # 2. 兼容 reasoning_content 字段 (某些 API 格式)
        reasoning_field = getattr(message, 'reasoning_content', None)
        if reasoning_field:
            # print(f"⚠️ [{func_name}] Warning: 'content' is None, falling back to 'reasoning_content'.")
            return reasoning_field.strip()

        print(f"⚠️ [{func_name}] Warning: API returned None for both content and reasoning. Returning empty string.")
        return ""
        
    except Exception as e:
        print(f"❌ [{func_name}] Error parsing response: {e}")
        return ""


def _resolve_system_prompt(default_prompt, kwargs):
    if 'sys_prompt' in kwargs:
        return kwargs['sys_prompt']
    return default_prompt

def _make_api_call_with_retry(call_func, max_retries=5):
    """
    🔥 核心重试逻辑：遇到 429 限流错误时，指数级等待并重试
    """
    for attempt in range(max_retries):
        try:
            return call_func()  # 尝试执行传入的 API 调用函数
        except Exception as e:
            error_msg = str(e)
            # 如果是限流 (429) 或 服务过载 (503)
            if "429" in error_msg or "Too many requests" in error_msg or "503" in error_msg:
                # 指数退避：2s, 4s, 8s... + 随机抖动
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ 触发限流 (429/503)。正在等待 {wait_time:.1f}秒后重试... (尝试 {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # 其他错误（如 Key 错误、参数错误）直接抛出，不重试
                raise e
    
    raise Exception(f"❌ 重试 {max_retries} 次后依然失败，请检查 API 额度或网络。")

# =========================================================================
# 🤖 模型调用函数 (已接入自动重试)
# =========================================================================

def call_deepseek_llm(prompt: str, system_prompt: str = "Research Assistant", temperature: float = 0.05, max_tokens: int = 8192, **kwargs) -> str:
    if not _check_client(): return "❌ 系统未配置 API Key，请在侧边栏输入。"
    
    final_sys_prompt = _resolve_system_prompt(system_prompt, kwargs)
    
    # 1. 定义动作：怎么调 API
    def _api_call():
        return client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": final_sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # 2. 执行动作：交给重试控制器去跑
    try:
        response = _make_api_call_with_retry(_api_call)
        return _safe_get_content(response, "call_deepseek_llm")
    except Exception as e:
        print(f"❌ LLM Call Failed: {e}")
        return f"❌ API Error: {e}"

def call_deepseek_llm_Intent(prompt: str, system_prompt: str = "Intent Classifier", temperature: float = 0.1, max_tokens: int = 1024, **kwargs) -> str:
    if not _check_client(): return "Error: No API Key"
    
    final_sys_prompt = _resolve_system_prompt(system_prompt, kwargs)
    
    def _api_call():
        return client.chat.completions.create(
            model=INTENT_MODEL,   
            messages=[
                {"role": "system", "content": final_sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        response = _make_api_call_with_retry(_api_call)
        return _safe_get_content(response, "Intent")
    except Exception as e:
        print(f"❌ Intent Failed: {e}")
        return ""

def call_deepseek_llm_Design(prompt: str, system_prompt: str = "Designer", temperature: float = 0.5, max_tokens: int = 8192, **kwargs) -> str:
    if not _check_client(): return "Error: No API Key"

    final_sys_prompt = _resolve_system_prompt(system_prompt, kwargs)
    
    def _api_call():
        return client.chat.completions.create(
            model=DESIGN_MODEL,    
            messages=[
                {"role": "system", "content": final_sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        response = _make_api_call_with_retry(_api_call)
        return _safe_get_content(response, "Design")
    except Exception as e:
        print(f"❌ Design Failed: {e}")
        return ""

def call_deepseek_llm_Score(prompt: str, system_prompt: str = "Reviewer", temperature: float = 0.0,  max_tokens: int = 8192, **kwargs) -> str:
    if not _check_client(): return "Error: No API Key"

    final_sys_prompt = _resolve_system_prompt(system_prompt, kwargs)
    
    def _api_call():
        return client.chat.completions.create(
            model=SCORE_MODEL,      
            messages=[
                {"role": "system", "content": final_sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        response = _make_api_call_with_retry(_api_call)
        return _safe_get_content(response, "Score")
    except Exception as e:
        print(f"❌ Score Failed: {e}")
        return ""

def call_deepseek_llm_Planner_Module(prompt: str, system_prompt: str = "Planner", temperature: float = 0.0,  max_tokens: int = 8192, **kwargs) -> str:
    if not _check_client(): return "Error: No API Key"

    final_sys_prompt = _resolve_system_prompt(system_prompt, kwargs)
    
    def _api_call():
        return client.chat.completions.create(
            model=PLANNER_Module_MODEL,   
            messages=[
                {"role": "system", "content": final_sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        response = _make_api_call_with_retry(_api_call)
        return _safe_get_content(response, "Planner")
    except Exception as e:
        print(f"❌ Planner Failed: {e}")
        return ""

def call_deepseek_llm_Report(prompt: str, system_prompt: str = "Academic Paper Writer", temperature: float = 0.15, max_tokens: int = 8192, **kwargs) -> str:
    if not _check_client(): return "Error: No API Key"

    final_sys_prompt = _resolve_system_prompt(system_prompt, kwargs)
    
    # 定义调用动作
    def _api_call():
        return client.chat.completions.create(
            model=REPORT_MODEL, # 通常写作用主模型即可，也可以换成专门的 Writer 模型
            messages=[
                {"role": "system", "content": final_sys_prompt},
                {"role": "user", "content": prompt},
            ],
            # 🔥 关键配置：写论文要严谨(低温)且长(大Token)
            temperature=temperature,
            max_tokens=max_tokens,
        )
    try:
        # 写论文耗时久，容易超时，底层重试机制非常重要
        response = _make_api_call_with_retry(_api_call)
        return _safe_get_content(response, "Report_Writer")
    except Exception as e:
        print(f"❌ Report Generation Failed: {e}")
        return ""

def get_embedding_via_api(text):
    if not _check_client(): return np.zeros(1024) #1024glm；4096doubao
    
    # 嵌入也需要重试保护
    def _api_call():
        return client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )

    try:
        response = _make_api_call_with_retry(_api_call)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"❌ Embedding Failed: {e}")
        return np.zeros(1024)
