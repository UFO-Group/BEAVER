# Agent/Utils/data_visualizer.py

import os
import matplotlib
# 🔥🔥🔥 [核心修复] 强制使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from rich.console import Console

# ❌ 彻底移除 LLM 和 文本解析工具的引用
# 只保留绘图所需的库
console = Console()

def create_comparison_chart(results_data, save_dir):
    """
    【离线 4维极速版】
    1. 接收 Main 程序传来的 'scores_dict' (包含 Feasibility, Predictability, Performance, Innovation)。
    2. 使用 Matplotlib 绘制分组柱状图。
    3. ❌ 不调用 LLM，0 Token 消耗，速度毫秒级。
    """
    console.print("\n[bold magenta]📊 Visualizing Data (4-Dimensions Local)...[/bold magenta]")

    if not results_data:
        console.print("[yellow]⚠️ No data for visualization.[/yellow]")
        return None

    try:
        # 1. 准备数据容器
        ids = []
        titles = []
        
        # 🔥 [修改点 1] 准备四个列表用于 matplotlib 分组绘制
        feasibility_list = []
        predictability_list = []
        performance_list = []
        innovation_list = [] # 新增

        for item in results_data:
            ids.append(item.get('id', 'Unknown'))
            titles.append(item.get('title', 'Untitled'))
            
            # 🔥 直接读取 Main 传过来的字典
            default_score = item.get('score', 0)
            s_dict = item.get('scores_dict', {})
            
            # 获取各维度分数
            feasibility_list.append(s_dict.get("Feasibility", default_score))
            predictability_list.append(s_dict.get("Predictability", default_score))
            performance_list.append(s_dict.get("Performance", default_score))
            innovation_list.append(s_dict.get("Innovation", default_score)) # 新增

        # 2. 设置绘图参数
        x = np.arange(len(ids))  # 想法的数量
        
        # 🔥 [修改点 2] 调整柱子宽度，因为现在有4根柱子，要细一点
        total_width = 0.8 # 总宽度
        n_bars = 4 # 柱子数量
        width = total_width / n_bars
        
        # 字体设置
        try:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman', 'Arial', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

        fig, ax = plt.subplots(figsize=(12, 6))

        # 3. 绘制四组柱子
        # 计算偏移量：
        # 第1根: x - 1.5*width
        # 第2根: x - 0.5*width
        # 第3根: x + 0.5*width
        # 第4根: x + 1.5*width
        
        # 颜色方案 (Tableau Palette): Blue, Orange, Green, Red
        c1, c2, c3, c4 = '#4E79A7', '#F28E2B', '#59A14F', '#E15759'

        rects1 = ax.bar(x - 1.5*width, feasibility_list, width, label='Feasibility', color=c1, edgecolor='black', alpha=0.9)
        rects2 = ax.bar(x - 0.5*width, predictability_list, width, label='Predictability', color=c2, edgecolor='black', alpha=0.9)
        rects3 = ax.bar(x + 0.5*width, performance_list, width, label='Performance', color=c3, edgecolor='black', alpha=0.9)
        rects4 = ax.bar(x + 1.5*width, innovation_list, width, label='Innovation', color=c4, edgecolor='black', alpha=0.9, hatch='//') # 给创新性加个斜线纹理突出一下

        # 4. 图表修饰
        ax.set_ylabel('Score (0-100)', fontsize=12)
        ax.set_title('Multi-Dimensional Analysis (Includes Innovation)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        
        # X轴标签优化
        x_labels = []
        for i, title in enumerate(titles):
            short_title = textwrap.shorten(title, width=15, placeholder="...")
            x_labels.append(f"{ids[i]}\n{short_title}")
        ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
        
        # 图例放下方
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncols=4, fontsize=10) 
        
        # 设置 Y 轴范围
        ax.set_ylim(0, 119) 

        # 5. 柱子顶部标数值
        ax.bar_label(rects1, padding=3, fontsize=8)
        ax.bar_label(rects2, padding=3, fontsize=8)
        ax.bar_label(rects3, padding=3, fontsize=8)
        ax.bar_label(rects4, padding=3, fontsize=8)

        # 添加网格
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)

        # 6. 保存图片
        chart_filename = "comparison_chart.png" # 名字简化一点
        chart_path = os.path.join(save_dir, chart_filename)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✔ Chart generated: {chart_path}[/green]")
        return chart_path

    except Exception as e:
        console.print(f"[red]⚠️ Visualization failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None