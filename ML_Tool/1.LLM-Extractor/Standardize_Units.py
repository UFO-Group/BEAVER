import os
import sys
import glob
import pandas as pd
import re
from datetime import datetime

# ================= Core Cleaning Functions =================

def convert_mech(val):
    """Mechanical property unit conversion -> MPa"""
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()

    # 1. Preprocess special characters
    s = s.replace("N/mm²", "MPa").replace("N/mm2", "MPa")
    s = s.replace("MN/m²", "MPa").replace("N/m²", "Pa").replace("N/m2", "Pa")

    # 2. Extract number (prioritize the first appearing value, ignore ranges)
    # The new version extraction script might output "10-20 MPa", here we extract "10"
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m:
        return pd.NA
    num = float(m.group())
    unit = s.lower()

    # 3. Unit conversion
    if 'gpa' in unit:
        return num * 1000
    elif 'mpa' in unit:
        return num
    elif 'kpa' in unit:
        return num / 1000
    elif 'pa' in unit:
        return num / 1e6
    elif 'cn/dtex' in unit:
        return num * 0.088055
    elif 'cn/tex' in unit:
        return num * 0.1
    elif 'n/tex' in unit:
        return num * 100
    elif 'dyn/cm2' in unit or 'dyn/cm²' in unit:
        return num / 1e7
    else:
        return num

def convert_impact(val):
    """Impact strength unit conversion -> kJ/m²"""
    if pd.isna(val):
        return pd.NA
    s = str(val).lower().strip()

    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m:
        return pd.NA
    num = float(m.group())

    if 'j/cm²' in s or 'j/cm2' in s:
        return num * 10
    elif 'kj/m²' in s or 'kj/m2' in s:
        return num
    elif 'j/m²' in s or 'j/m2' in s:
        return num / 1000
    else:
        return num

def convert_temp(val):
    """Temperature conversion K -> °C"""
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()

    # Remove plain text unit markers to prevent interference with number extraction
    s_clean = re.sub(r'\s*(°\s*[CFK]|℃|[CFK])\s*', '', s, flags=re.IGNORECASE)
    m = re.search(r'-?\d+(?:\.\d+)?', s_clean)
    if not m:
        return pd.NA
    num = float(m.group())

    # Only subtract 273.15 if K is explicitly detected (and not in the preceding text)
    # Simple strategy: If the original string contains K (case-insensitive) and value > 200 (to prevent misidentifying low temps), consider it Kelvin
    if re.search(r'(?<!°)\bK\b', str(val), re.IGNORECASE) and num > 200:
        return num - 273.15
    else:
        return num

def extract_numeric(val):
    """Extract percentage value"""
    s = str(val)
    m = re.match(r'(-?\d+(?:\.\d+)?)', s.strip())
    return float(m.group(1)) if m else pd.NA

def should_remove(val: str):
    """Clean invalid data such as years or months"""
    val = str(val).strip()
    if re.fullmatch(r"(19|20)\d{2}", val):
        return True
    if re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', val, re.IGNORECASE):
        return True
    if re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', val, re.IGNORECASE):
        return True
    return False

def run_standardization(input_path, output_path):
    print("="*60)
    print("Starting Unit Standardization")
    print(f"📁 Reading: {input_path}")
    
    if not os.path.exists(input_path):
        print("❌ Error: File does not exist")
        return

    try:
        df = pd.read_csv(input_path, dtype=str)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # 1. Remove invalid cells
    removed_count = 0
    for col in df.columns:
        for i, val in df[col].items():
            if pd.notna(val) and should_remove(val):
                removed_count += 1
                df.at[i, col] = pd.NA
    
    # 2. Field definitions (match output of Table_Generation.py)
    mechanical_mpa_fields = ["Tensile Strength", "Young's Modulus", "Flexural Modulus"]
    percent_fields = ["Elongation at Break"]
    temperature_k_fields = ["Glass Transition", "Melting Point"]

    # 3. Execute conversion
    # Auto-detect Impact Strength
    impact_fields = [col for col in df.columns if re.search(r'impact\s*strength', col, re.IGNORECASE)]
    for col in impact_fields:
        df[col] = df[col].apply(convert_impact)

    # Mechanical properties
    for col in mechanical_mpa_fields:
        if col in df.columns:
            df[col] = df[col].apply(convert_mech)

    # Percentage
    for col in percent_fields:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    # Temperature
    for col in temperature_k_fields:
        if col in df.columns:
            df[col] = df[col].apply(convert_temp)

    # 4. Rename fields (add units)
    rename_map = {
        "Tensile Strength": "Tensile Strength (MPa)",
        "Young's Modulus": "Young's Modulus (MPa)",
        "Flexural Modulus": "Flexural Modulus (MPa)",
        "Elongation at Break": "Elongation at Break (%)",
        "Glass Transition": "Glass Transition Temperature (°C)",
        "Melting Point": "Melting Temperature (°C)",
        # If new script has Impact Strength, renaming is also recommended, although it is dynamically detected above
        "Impact Strength": "Impact Strength (kJ/m²)"
    }
    df.rename(columns=rename_map, inplace=True)

    # 5. Save
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Processing complete: {len(df)} rows")
    print(f"🧹 Cleaned invalid cells: {removed_count} count")
    print(f"💾 Saved to: {output_path}")
    print("="*60)

# ================= Main Execution Block (Pipeline Adapter) =================

def find_input_csv_fallback():
    """Fallback: If not called via pipeline, try to find automatically"""
    search_locations = [
        "Data_Extraction/*.csv",
        "Data_Extraction/*Extraction_Result*.csv",
        "Data/Processed_Results/*.csv"
    ]
    for pattern in search_locations:
        csv_files = glob.glob(pattern, recursive=True)
        if csv_files:
            return csv_files[0] # Return the latest or the first one
    return None

if __name__ == "__main__":

    input_csv  = r"Your_split_pdfs_Path_Extraction_Result_1.csv"
    output_csv = r"Your_split_pdfs_Path_Extraction_Result_Standardize_1.csv"

    # Logic check: Is it a Pipeline call or manual run?
    if "Your_split_pdfs_Path" in input_csv:
        # Entering here means Pipeline did not replace paths, currently in manual run mode
        print("⚠️ Manual run mode detected (using default path search logic)")
        
        found_input = find_input_csv_fallback()
        if not found_input:
            print("❌ Input CSV not found, please check file location or manually modify path in code.")
            sys.exit(1)
            
        input_csv = found_input
        # Generate a timestamped output file in manual mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"Data/Processed_Results/Standardized_Manual_{timestamp}.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Run main logic
    run_standardization(input_csv, output_csv)