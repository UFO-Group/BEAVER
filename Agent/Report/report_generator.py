import sys
import os
import re
import io
import concurrent.futures
from rich.console import Console

# =========================================================================
# 📦 依赖导入与环境配置
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from paper_writer import generate_single_paper
from docx_engine import save_text_to_docx
# NOTE:
# generate_single_paper / generate_comparison_report 保留为 legacy packaging fallback。
# 当前 design 主链的最终报告已在 plan_executor -> reason_over_evidence 中完成生成。
from Agent.Utils.file_utils import save_step_result

# 1. 尝试导入 DeepSeek 客户端
try:
    from Agent.Agent_Config.deepseek_client import call_deepseek_llm
    from Agent.Utils.file_utils import save_step_result
except ImportError:
    # 用于防止单独运行时报错的 fallback
    pass

# 2. 尝试导入 python-docx 及其组件
try:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("⚠️ Warning: python-docx not installed. Word files will not be generated.")

# 3. 🔥 [新增] 尝试导入 docx2pdf (用于转 PDF)
# 注意：此库依赖系统安装 Microsoft Word (Windows/macOS)
try:
    from docx2pdf import convert as convert_to_pdf
    HAS_PDF_CONVERTER = True
except ImportError:
    HAS_PDF_CONVERTER = False
    print("⚠️ Warning: docx2pdf not installed. PDF will not be generated.")

# 4. 🔥 [新增] 尝试导入 matplotlib (用于渲染公式)
try:
    import matplotlib
    # 强制设置非交互后端 'Agg'，防止多线程绘图时报错或弹出窗口
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ Warning: matplotlib not installed. Formulas will render as raw text.")

console = Console()

def process_single_report_task(args):
    raise RuntimeError(
        "process_single_report_task() is deprecated. "
        "Design reports must be generated only via "
        "plan_executor -> reason_over_evidence."
    )

def generate_comparison_report(user_query: str, results_data: list, save_path: str = None) -> str:
    raise RuntimeError(
        "generate_comparison_report() is deprecated. "
        "Use the current design main pipeline instead."
    )