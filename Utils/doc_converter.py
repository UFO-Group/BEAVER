import pypandoc
import os
from rich.console import Console

# ✅ 1. 强制指定 Pandoc 路径 (保留你之前测试成功的设置)
os.environ["PYPANDOC_PANDOC"] = r"C:\Program Files\Pandoc\pandoc.exe"

console = Console()

def convert_md_to_doc(md_content, output_path):
    """
    将 Markdown 转换为 Word (.docx)。
    自动修复图片路径问题，并强制输出为 docx 格式。
    """
    # ✅ 2. 强制把后缀从 .pdf 改为 .docx (避免 LaTeX 报错)
    if output_path.endswith('.pdf'):
        output_path = output_path.replace('.pdf', '.docx')
        
    # ✅ 3. 获取输出文件所在的目录 (关键修复！)
    # 解释：图片 (comparison_chart.png) 就在这个目录下。
    # 我们需要提取这个路径传给 Pandoc，否则它找不到图片。
    output_dir = os.path.dirname(os.path.abspath(output_path))

    console.print("\n[bold magenta]📄 Converting to Word (Docx)...[/bold magenta]")
    # console.print(f"[dim]🔍 Resource path set to: {output_dir}[/dim]") # 调试用，可注释

    try:        
        pypandoc.convert_text(
            md_content, 
            'docx',              # 目标格式 Word
            format='md', 
            outputfile=output_path,
            # 🔥 4. 关键参数：告诉 Pandoc 去哪里找图片
            extra_args=[f'--resource-path={output_dir}'] 
        )
        console.print(f"[green]✔ Document saved: {output_path}[/green]")
        return True

    except OSError:
        console.print("[red]❌ Pandoc not found or path incorrect.[/red]")
        console.print(f"[dim]Current path setting: {os.environ.get('PYPANDOC_PANDOC')}[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]⚠️ Conversion failed: {e}[/red]")
        return False