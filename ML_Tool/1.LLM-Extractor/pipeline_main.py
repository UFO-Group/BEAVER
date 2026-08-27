#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import traceback
from datetime import datetime

# ================== Core Helper Functions ==================

def run_script_in_memory(script_path, replacements=None):
    """
    Reads the script, replaces configurations, and executes it directly in memory 
    without generating temporary files.
    """
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found {script_path}")
        return False

    try:
        # 1. Read code
        with open(script_path, 'r', encoding='utf-8') as f:
            code_content = f.read()

        # 2. Replace path configurations in memory
        if replacements:
            for old_str, new_str in replacements.items():
                # Note: We use raw strings in replacements to avoid escape char issues
                code_content = code_content.replace(old_str, new_str)

        # 3. Create execution context
        exec_globals = {'__name__': '__main__'}

        # 4. Execute
        exec(code_content, exec_globals)
        return True

    except Exception as e:
        print(f"❌ Execution of {script_path} failed: {e}")
        traceback.print_exc()
        return False

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {description}")
    print(f"{'='*60}")

# ================== Business Logic ==================

def load_config():
    config = {
        "pdf_raw_dir": "Data",
        "pdf_split_dir": "Data/Data_split",
        "output_dir": "Data/Processed_Results",
        "dict_dir": "dict",  # 新增字典路径配置
        "chunk_size": 3
    }
    
    config_file = "pipeline_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
            print(f"Loading configuration from {config_file}")
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
    return config

def create_directories(config):
    dirs = [
        config['pdf_raw_dir'],
        config['pdf_split_dir'],
        config['output_dir']
    ]
    for d in dirs:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"📁 Creating directory: {d}")

def check_pdfs(config):
    print_step(1, "Checking PDF Files")
    input_dir = config['pdf_raw_dir']
    
    if not os.path.exists(input_dir):
        print(f"❌ Directory does not exist: {input_dir}")
        return False
        
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf') and not f.startswith('._')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return False
        
    print(f"📄 Found {len(pdf_files)} PDF files")
    return True

def step_split_pdf(config):
    print_step(2, "Splitting PDFs")
    
    replacements = {
        'input_folder = r"D:\\\\FXR\\\\PDF_raw"': f'input_folder = r"{config["pdf_raw_dir"]}"',
        'output_base_folder = r"D:\\\\FXR\\\\PDF_split_5pages"': f'output_base_folder = r"{config["pdf_split_dir"]}"',
        'chunk_size = 5': f'chunk_size = {config["chunk_size"]}'
    }
    
    return run_script_in_memory("Split_pdf.py", replacements)

def step_data_extraction(config):
    print_step(3, "Extracting Data")
    
    possible_names = [
        "Data_Extraction/main_PDF_Youngsmodulus.py",
        "main_PDF_Youngsmodulus.py"
    ]
    script_path = next((name for name in possible_names if os.path.exists(name)), None)
    
    if not script_path:
        print("❌ Data extraction script not found")
        return False

    replacements = {
        'INPUT_FOLDER = r"Your_split_pdfs_Path"': f'INPUT_FOLDER = r"{config["pdf_split_dir"]}"'
    }

    sys.path.insert(0, os.path.abspath("Data_Extraction"))
    sys.path.insert(0, os.path.abspath("."))
    
    print("🚀 Starting data extraction...")
    result = run_script_in_memory(script_path, replacements)
    
    sys.path.pop(0)
    sys.path.pop(0)
    
    return result

def step_table_generation(config):
    print_step(4, "Table Generation")
    output_csv = os.path.join(config['output_dir'], "Table_Result.csv")
    
    replacements = {
        'root = r"Your_split_pdfs_Path"': f'root = r"{config["pdf_split_dir"]}"',
        'out_csv = os.path.join(root, "Extraction_Result.csv")': f'out_csv = r"{output_csv}"'
    }
    
    if run_script_in_memory("Table_Generation.py", replacements):
        if os.path.exists(output_csv):
            print(f"✅ Result generated: {output_csv}")
            return output_csv
    return None

def step_standardization(config, input_csv):
    print_step(5, "Unit Standardization")
    
    if not input_csv: 
        return None

    output_csv = os.path.join(config['output_dir'], "Standardized_Result.csv")
    
    replacements = {
        'input_csv  = r"Your_split_pdfs_Path_Extraction_Result_1.csv"': f'input_csv  = r"{input_csv}"',
        'output_csv = r"Your_split_pdfs_Path_Extraction_Result_Standardize_1.csv"': f'output_csv = r"{output_csv}"'
    }
    
    if run_script_in_memory("Standardize_Units.py", replacements):
        if os.path.exists(output_csv):
            print(f"✅ Standardized Result: {output_csv}")
            return output_csv
    return None

def step_classification(config, input_csv):
    """
    新增步骤：调用归类脚本
    """
    print_step(6, "Name Classification")
    
    if not input_csv:
        print("❌ Skip classification: No input CSV.")
        return None

    # 定义最终输出文件名
    final_output_csv = os.path.join(config['output_dir'], "Final_Classified_Result.csv")
    
    # 获取字典的绝对路径 (防止相对路径在 exec 中出错)
    dict_abs_path = os.path.abspath(config['dict_dir'])

    # 替换 Name_Classification.py 中的配置变量
    # 注意：这里使用的是 Path("...") 的字符串替换形式
    replacements = {
        'DICT_DIR = Path("dict")': f'DICT_DIR = Path(r"{dict_abs_path}")',
        'INPUT_CSV = Path(r"Data/Processed_Results/Standardized_Result.csv")': f'INPUT_CSV = Path(r"{input_csv}")',
        'OUTPUT_CSV = Path(r"Data/Processed_Results/Final_Classified_Result.csv")': f'OUTPUT_CSV = Path(r"{final_output_csv}")'
    }

    if run_script_in_memory("Name_Classification.py", replacements):
        if os.path.exists(final_output_csv):
            print(f"✅ Final Classified Result: {final_output_csv}")
            return final_output_csv
    
    return None

def main():
    print("="*60)
    print(f"PDF Processing Pipeline (Start: {datetime.now().strftime('%H:%M:%S')})")
    print("="*60)
    
    config = load_config()
    create_directories(config)
    
    # 1. Check
    if not check_pdfs(config): return
    # 2. Split
    if not step_split_pdf(config): return
    # 3. Extract
    if not step_data_extraction(config): return
    
    # 4. Table Gen
    table_csv = step_table_generation(config)
    if not table_csv: return
    
    # 5. Standardize
    std_csv = step_standardization(config, table_csv)
    if not std_csv: return

    # 6. Classification (New Step)
    final_csv = step_classification(config, std_csv)
    
    print("\n" + "="*60)
    print("🎉 All done!")
    if final_csv:
        print(f"📄 Final Output: {final_csv}")
    print("="*60)

if __name__ == "__main__":
    main()