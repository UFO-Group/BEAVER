import pandas as pd
import numpy as np
import json
import os

# ===========================
#  数据库路径
# ===========================
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_ROOT = os.path.dirname(AGENT_ROOT)
MECH_DB_PATH = os.path.join(
    SUBMISSION_ROOT,
    "ML_Tool",
    "3.ML_Tool",
    "Prediction-AB.csv",
)

# ===========================
#  Polymer / Additive 字段 (保持不变)
# ===========================
POLYMER_NAME_COLUMNS = ["Polymer A Name", "Polymer B Name", "Polymer C Name", "Polymer D Name", "Polymer E Name"]
POLYMER_TYPE_COLUMNS = ["Polymer A Type", "Polymer B Type", "Polymer C Type", "Polymer D Type", "Polymer E Type"]
POLYMER_ROLE_COLUMNS = ["Polymer A Role", "Polymer B Role", "Polymer C Role", "Polymer D Role", "Polymer E Role"]

ADDITIVE_NAME_COLUMNS = ["Material A Name", "Material B Name", "Material C Name", "Material D Name", "Material E Name"]
ADDITIVE_TYPE_COLUMNS = ["Material A Type", "Material B Type", "Material C Type", "Material D Type", "Material E Type"]
ADDITIVE_ROLE_COLUMNS = ["Material A Role", "Material B Role", "Material C Role", "Material D Role", "Material E Role"]

# ===========================
#  全部机械性能字段（严格限定你指定的5个）
# ===========================
MECH_PROPERTY_COLUMNS = {
    "tensile_strength":          "Tensile Strength (MPa)",
    "elongation_at_break":       "Elongation at Break (%)",
    "youngs_modulus":            "Young's Modulus (kPa)",            # 注意单位是 kPa
    "glass_transition":          "Glass Transition Temperature (°C)", # 符号已适配
    "melting_temperature":       "Melting Temperature (°C)",          # 符号已适配
}

# 别名映射 (容错用)
KEY_ALIAS_MAP = {
    "min_tensile_strength_MPa":      "min_tensile_strength",
    "max_tensile_strength_MPa":      "max_tensile_strength",
    "min_elongation_at_break_%":     "min_elongation_at_break",
    "max_elongation_at_break_%":     "max_elongation_at_break",
    "min_youngs_modulus_kPa":        "min_youngs_modulus",
    "max_youngs_modulus_kPa":        "max_youngs_modulus",
    "min_glass_transition_C":        "min_glass_transition",
    "max_glass_transition_C":        "max_glass_transition",
    "min_melting_temperature_C":     "min_melting_temperature",
    "max_melting_temperature_C":     "max_melting_temperature",
}

def normalize_inputs(inputs: dict) -> dict:
    """标准化输入键名"""
    norm = {}
    for k, v in inputs.items():
        if k in KEY_ALIAS_MAP:
            norm[KEY_ALIAS_MAP[k]] = v
        else:
            norm[k] = v
    return norm

def to_float_safe(series):
    return pd.to_numeric(series, errors="coerce")

# ==========================================================
#  ⭐ 主函数：支持 Min/Max/Target/Raw + 智能排序
# ==========================================================
def run_mech_table_query(inputs: dict, max_rows: int = None):
    
    inputs = normalize_inputs(inputs)

    # 1. 读取数据库 (多编码自动尝试)
    if not os.path.exists(MECH_DB_PATH):
        print(f"❌ Error: Database not found at {MECH_DB_PATH}")
        return []

    df = None
    # 定义编码尝试列表：优先中文环境(gbk/gb18030)，其次通用(utf-8)，最后兜底(latin1)
    encodings_to_try = ['gbk', 'utf-8', 'utf-8-sig', 'gb18030', 'gb2312', 'latin1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            # print(f"[Debug] Trying to read CSV with encoding: {encoding}...") 
            df = pd.read_csv(MECH_DB_PATH, low_memory=False, encoding=encoding)
            # 读成功了，直接跳出循环
            # print(f"✅ Successfully read CSV with encoding: {encoding}")
            break 
        except UnicodeDecodeError:
            continue # 编码不对，试下一个
        except Exception as e:
            # 如果是路径错误或其他非编码错误，没必要继续试了
            print(f"❌ Critical Error reading CSV: {e}")
            return []

    # 如果试了一圈还是 None，说明彻底失败
    if df is None:
        print(f"❌ Failed to read CSV. Tried encodings: {encodings_to_try}")
        return []
        
    df = df.replace({np.nan: None})

    # 2. 预处理数值列 (只处理那5个列)
    for key, col_name in MECH_PROPERTY_COLUMNS.items():
        if col_name in df.columns:
            df[col_name] = to_float_safe(df[col_name])

    # 3. 文本筛选
    text_filters = [
        ("polymer", POLYMER_NAME_COLUMNS),
        ("polymer_type", POLYMER_TYPE_COLUMNS),
        ("polymer_role", POLYMER_ROLE_COLUMNS),
        ("additive", ADDITIVE_NAME_COLUMNS),
        ("additive_type", ADDITIVE_TYPE_COLUMNS),
        ("additive_role", ADDITIVE_ROLE_COLUMNS)
    ]
    
    for input_key, cols in text_filters:
        if input_key in inputs:
            key = str(inputs[input_key]).lower()
            mask = np.zeros(len(df), dtype=bool)
            for col in cols:
                if col in df.columns:
                    mask |= df[col].fillna("").astype(str).str.lower().str.contains(key, regex=False)
            df = df[mask]
            
    # 特殊结构筛选
    if "structure_type" in inputs and "Polymer Structure Type" in df.columns:
        k = str(inputs["structure_type"]).lower()
        df = df[df["Polymer Structure Type"].fillna("").astype(str).str.lower().str.contains(k, regex=False)]
    
    if "filled_with" in inputs and "Copolymerized/Blended/Crosslinked/Filled With" in df.columns:
        k = str(inputs["filled_with"]).lower()
        df = df[df["Copolymerized/Blended/Crosslinked/Filled With"].fillna("").astype(str).str.lower().str.contains(k, regex=False)]

    # 4. 🔥 数值筛选 (核心逻辑)
    for simple_key, col_name in MECH_PROPERTY_COLUMNS.items():
        if col_name not in df.columns:
            continue
            
        # A. Min (>): min_tensile_strength
        if f"min_{simple_key}" in inputs:
            try:
                val = float(inputs[f"min_{simple_key}"])
                df = df[df[col_name] >= val]
            except: pass

        # B. Max (<): max_tensile_strength
        if f"max_{simple_key}" in inputs:
            try:
                val = float(inputs[f"max_{simple_key}"])
                df = df[df[col_name] <= val]
            except: pass

        # C. Target (≈): target_glass_transition (±10%)
        if f"target_{simple_key}" in inputs:
            try:
                val = float(inputs[f"target_{simple_key}"])
                df = df[(df[col_name] >= val * 0.9) & (df[col_name] <= val * 1.1)]
            except: pass
            
        # D. Raw (>): tensile_strength (默认当做最小值处理)
        if simple_key in inputs:
            try:
                val = float(inputs[simple_key])
                df = df[df[col_name] >= val]
            except: pass

    # 5. 🔥 智能排序
    sort_col = None
    ascending = False # 默认降序(越大越好)

    # 优先：如果有 Target，按“距离目标最近”排
    for key, col_name in MECH_PROPERTY_COLUMNS.items():
        if f"target_{key}" in inputs and col_name in df.columns:
            try:
                t_val = float(inputs[f"target_{key}"])
                df["_diff"] = abs(df[col_name] - t_val)
                sort_col = "_diff"
                ascending = True # 差值越小越好
                break
            except: pass
    
    # 次选：按 Min/Raw 值降序 (最强的排前面)
    if not sort_col:
        for key, col_name in MECH_PROPERTY_COLUMNS.items():
            if (f"min_{key}" in inputs or key in inputs) and col_name in df.columns:
                sort_col = col_name
                ascending = False
                break
                
    if sort_col:
        df = df.sort_values(by=sort_col, ascending=ascending)
        if sort_col == "_diff": df = df.drop(columns=["_diff"])

    # 6. 返回结果
    if max_rows is not None and isinstance(max_rows, int):
        df = df.head(max_rows)
    
    return df.to_dict(orient="records")
