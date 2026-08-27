import re
import io

# === 增强：在模块内部处理依赖检测，确保自包含性 ===
try:
    import matplotlib
    # 强制设置非交互后端 'Agg'，防止多线程绘图时报错或弹出窗口
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # === 核心修改：配置字体参数，防止乱码 ===
    plt.rcParams['mathtext.fontset'] = 'cm'  # 使用 Computer Modern (经典 LaTeX 字体)
    # 如果想用更粗一点的字体，可以使用 'stix'
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ Warning: matplotlib not installed. Formulas will render as raw text.")

# =========================================================================
# 🧹 工具函数：清洗行内 LaTeX (让正文看起来更正常)
# =========================================================================
def clean_inline_latex_to_unicode(text):
    """
    将文本中常见的 LaTeX 符号转换为 Unicode 字符，
    防止在 Word 正文中出现 \Delta, \text{}, \geq 等乱码。
    """
    if not text: return text
    
    # 1. 移除 \text{...} 包装，只保留内容
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    
    # 2. 常用数学符号映射表
    replacements = {
        r'\geq': '≥', r'\leq': '≤', r'\Delta': 'Δ', r'\approx': '≈',
        r'\times': '×', r'\cdot': '·', r'\rightarrow': '→', r'\leftarrow': '←',
        r'\pm': '±', r'\circ': '°', r'\%': '%', r'\_': '_',
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\mu': 'μ',
        r'\sigma': 'σ', r'\epsilon': 'ε', r'\neq': '≠', r'\infty': '∞'
    }
    
    for latex, char in replacements.items():
        text = text.replace(latex, char)
    
    return text

# =========================================================================
# 🧮 辅助工具：LaTeX -> 图片流 (优化版 - 紧包裹模式)
# =========================================================================
def latex_to_image_stream(latex_str):
    """
    将 LaTeX 字符串渲染为内存中的图片流 (BytesIO)。
    """
    if not HAS_MATPLOTLIB:
        return None
    
    try:
        # 1. 清洗：移除包裹符
        clean_latex = latex_str.strip().replace('\\[', '').replace('\\]', '').replace('$$', '')
        
        # === 核心修改：过滤中文 ===
        # Matplotlib 的数学模式不支持中文，如果 LLM 生成了中文说明，会导致渲染崩溃
        # 这里只保留 ASCII 字符和常用希腊字母
        clean_latex = "".join([c for c in clean_latex if ord(c) < 128 or c in 'Δαβγμσε∞≈≠≤≥±×·°'])
        
        if not clean_latex.strip():
            return None

        # Matplotlib 需要 $ 包裹来识别 math mode
        render_text = f"${clean_latex}$"

        # === 核心修改：采用紧包裹策略消除空白 ===
        # 1. 初始画布设为极小，完全依赖 bbox_inches='tight' 来扩展
        # 提高 DPI 到 600 以保证清晰度
        fig = plt.figure(figsize=(0.1, 0.1), dpi=600)
        
        # 2. 使用固定的较大字号 (配合高 DPI)
        plt.text(0.5, 0.5, render_text, fontsize=20, 
                 ha='center', va='center', color='black')
        
        # 3. 移除坐标轴
        plt.axis('off')
        
        # 4. 保存到内存 buffer
        buf = io.BytesIO()
        # bbox_inches='tight' 自动裁剪掉所有空白，只保留文字
        # pad_inches=0.02 留极小边距防止切掉角标
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, transparent=True)
        plt.close(fig)
        
        buf.seek(0)
        return buf
    except Exception:
        # 渲染失败时不报错，返回 None 让主程序回退到纯文本
        return None