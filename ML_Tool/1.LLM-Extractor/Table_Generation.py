import os
import re
import json
import csv
from collections import Counter

# ================= Configuration Area =================
# Modified FIELDS to include Description columns and Composition Overview
FIELDS = [
    "Source",
    "Material Name",
    "Composition_Overview",  # New: Stores ratio info like "40% PP, 30% WF..."
    
    # Base Polymers (A-E)
    "Polymer A Name", "Polymer A Type", "Polymer A Role",
    "Polymer B Name", "Polymer B Type", "Polymer B Role",
    "Polymer C Name", "Polymer C Type", "Polymer C Role",
    "Polymer D Name", "Polymer D Type", "Polymer D Role",
    "Polymer E Name", "Polymer E Type", "Polymer E Role",
    
    "Polymer Structure Type",
    "Copolymerized/Blended/Crosslinked/Filled With",
    
    # Additives / Other Materials (A-E)
    "Material A Name", "Material A Type", "Material A Role",
    "Material B Name", "Material B Type", "Material B Role",
    "Material C Name", "Material C Type", "Material C Role",
    "Material D Name", "Material D Type", "Material D Role",
    "Material E Name", "Material E Type", "Material E Role",
    
    # Mechanical Properties (Value + Description)
    "Tensile Strength", "Tensile Strength_Desc",
    "Elongation at Break", "Elongation at Break_Desc",
    "Young's Modulus", "Young's Modulus_Desc",
    "Flexural Modulus", "Flexural Modulus_Desc",
    "Flexural Strength", "Flexural Strength_Desc", # Added based on your JSON
    "Tensile Modulus", "Tensile Modulus_Desc",     # Added based on your JSON
    "Impact Strength", "Impact Strength_Desc",
    "Stress-Strain", "Stress-Strain_Desc",
    "Hardness", "Hardness_Desc",
    
    # Thermal
    "Glass Transition", 
    "Melting Point"
]

# ================= Helper Functions =================

def extract_json_blocks(content: str):
    """Robust extraction of JSON blocks from mixed text"""
    # 1. Try Markdown blocks
    blocks = re.findall(r"```json\s*(.*?)\s*```", content, flags=re.DOTALL)
    if blocks: 
        return [b.strip() for b in blocks]
    
    # 2. Try generic code blocks
    blocks = re.findall(r"```\s*(.*?)\s*```", content, flags=re.DOTALL)
    valid = []
    for b in blocks:
        if b.strip().startswith("[") or b.strip().startswith("{"):
            valid.append(b.strip())
    if valid: return valid

    # 3. Fallback: Regex after specific headers
    m = re.search(r"🧾\s*(?:提取结果|Extraction Result)：\s*([\s\S]+)", content)
    if m:
        raw = m.group(1).strip()
        # Simple heuristic to find the JSON bracket boundaries
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            return [raw[start:end+1]]
            
    return []

def normalize_mech_key(key: str) -> str:
    """Map JSON keys to standardized CSV Column names"""
    k = key.lower().strip().replace("_", " ")
    mapping = {
        "tensile strength": "Tensile Strength",
        "breaking strength": "Tensile Strength",
        "elongation at break": "Elongation at Break",
        "breaking elongation": "Elongation at Break",
        "young's modulus": "Young's Modulus",
        "youngs modulus": "Young's Modulus",
        "elastic modulus": "Young's Modulus",
        "flexural modulus": "Flexural Modulus",
        "flexural strength": "Flexural Strength",
        "tensile modulus": "Tensile Modulus",
        "impact strength": "Impact Strength",
        "impact toughness": "Impact Strength",
        "stress-strain": "Stress-Strain",
        "stress strain": "Stress-Strain",
        "hardness": "Hardness"
    }
    return mapping.get(k, None)

def summarize_field_coverage(rows):
    counter = Counter()
    total = len(rows)
    if total == 0:
        print("\n📊 No data rows extracted")
        return

    for row in rows:
        for field in FIELDS:
            if row.get(field, "") not in ["", None, "null"]:
                counter[field] += 1

    print("\n📊 Field Coverage Statistics (Top 20):")
    # Sort by defined order in FIELDS
    for field in FIELDS:
        count = counter.get(field, 0)
        if count > 0:
            print(f"{field:<45} : {count}/{total} ({count/total:.1%})")

def convert_all_output_folders(root_folder: str, output_csv_path: str):
    print(f"📂 Processing directory: {root_folder}")
    all_rows = []

    if not os.path.exists(root_folder):
        print(f"❌ Error: Input directory does not exist {root_folder}")
        return

    for dirpath, _, filenames in os.walk(root_folder):
        for fn in filenames:
            if not fn.lower().endswith(".txt"):
                continue
            
            file_path = os.path.join(dirpath, fn)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            blocks = extract_json_blocks(content)
            
            for idx, raw_json in enumerate(blocks, start=1):
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError:
                    continue
                
                if data is None: continue
                if isinstance(data, dict): data = [data] # Force list
                
                for j, item in enumerate(data):
                    if not isinstance(item, dict): continue

                    # === Initialize Row ===
                    row = {k: "" for k in FIELDS}
                    row["Source"] = f"{fn}_Part{idx}_Material{j+1}"
                    
                    # === 1. Basic Info ===
                    row["Material Name"] = item.get("Material Name", "")
                    row["Polymer Structure Type"] = item.get("Polymer Structure Type", "")
                    
                    # Handle Filled With List
                    filled = item.get("Copolymerized/Blended/Crosslinked/Filled With")
                    if isinstance(filled, list):
                        row["Copolymerized/Blended/Crosslinked/Filled With"] = ", ".join(map(str, filled))
                    elif filled:
                        row["Copolymerized/Blended/Crosslinked/Filled With"] = str(filled)

                    # === 2. Component Ratio (CRITICAL FIX) ===
                    # Extract the "Original" text or flatten the components dict
                    comp_ratio = item.get("Component Ratio")
                    if isinstance(comp_ratio, dict):
                        original = comp_ratio.get("Original")
                        if original:
                            row["Composition_Overview"] = original
                        else:
                            # Fallback: construct string from Components dict
                            comps = comp_ratio.get("Components", {})
                            if comps:
                                parts = [f"{k}:{v}" for k,v in comps.items()]
                                row["Composition_Overview"] = ", ".join(parts)

                    # === 3. Base Polymer Mapping (Sequential) ===
                    # Maps whatever keys exist (Polymer A, Matrix, etc.) to columns A, B, C...
                    base_polys = item.get("Base Polymer(s)")
                    if isinstance(base_polys, dict):
                        # Sort keys to ensure deterministic order (Polymer A, Polymer B...)
                        sorted_keys = sorted(base_polys.keys())
                        for i, p_key in enumerate(sorted_keys):
                            if i >= 5: break # Max 5 polymers
                            suffix = chr(65 + i) # A, B, C...
                            p_data = base_polys[p_key]
                            if isinstance(p_data, dict):
                                row[f"Polymer {suffix} Name"] = p_data.get("Name", "")
                                row[f"Polymer {suffix} Type"] = p_data.get("Type", "")
                                row[f"Polymer {suffix} Role"] = p_data.get("Role", "")

                    # === 4. Other Materials Mapping (Sequential) ===
                    # CRITICAL FIX: Maps dynamic keys like "MAPP", "TiO2" to Material A, Material B...
                    others = item.get("Other Material(Dopants, Additives or Modifiers)")
                    if isinstance(others, dict):
                        # We use values directly because keys might be "MAPP" or "IFR"
                        # Enumerating values ensures we catch them all regardless of key name
                        other_items = list(others.values()) 
                        for i, m_data in enumerate(other_items):
                            if i >= 5: break # Max 5 additives
                            suffix = chr(65 + i) # A, B, C...
                            if isinstance(m_data, dict):
                                row[f"Material {suffix} Name"] = m_data.get("Name", "")
                                row[f"Material {suffix} Type"] = m_data.get("Type", "")
                                row[f"Material {suffix} Role"] = m_data.get("Role", "")

                    # === 5. Mechanical Properties (Deep Extraction) ===
                    mech = item.get("Mechanical Properties")
                    if isinstance(mech, dict):
                        for m_key, m_val in mech.items():
                            std_col = normalize_mech_key(m_key)
                            if not std_col or std_col not in FIELDS:
                                continue
                            
                            if isinstance(m_val, dict):
                                # Extract Value
                                val = m_val.get("value")
                                unit = m_val.get("unit")
                                val_str = str(val) if val is not None else ""
                                if unit and val_str:
                                    val_str += f" {unit}"
                                row[std_col] = val_str
                                
                                # Extract Description (New Feature)
                                desc = m_val.get("description")
                                desc_col = f"{std_col}_Desc"
                                if desc and desc_col in FIELDS:
                                    row[desc_col] = desc

                    # === 6. Thermal Properties ===
                    tg = item.get("Glass Transition Temperature")
                    if isinstance(tg, dict):
                        v, u = tg.get("value"), tg.get("unit")
                        row["Glass Transition"] = f"{v} {u}".strip() if v else ""
                        
                    tm = item.get("Melting Temperature")
                    if isinstance(tm, dict):
                        v, u = tm.get("value"), tm.get("unit")
                        row["Melting Point"] = f"{v} {u}".strip() if v else ""

                    all_rows.append(row)

    # Write CSV
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as cf:
            writer = csv.DictWriter(cf, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\n✅ Extraction complete, total {len(all_rows)} records")
        print(f"📁 Result saved to: {output_csv_path}")
        summarize_field_coverage(all_rows)
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")

# ================= Main Execution Block =================
if __name__ == "__main__":
    # 1. Define placeholder paths
    root = r"Your_split_pdfs_Path"
    out_csv = os.path.join(root, "Extraction_Result.csv")
    
    # 2. Warning for manual run
    if "Your_split_pdfs_Path" in root:
        print("⚠️ Warning: Detected use of default placeholder path.")
        print("If manual testing, please modify the root path in the code.")
        # root = r"C:\Users\Admin\Desktop\Test_Data"  # Uncomment for manual testing
        # out_csv = r"C:\Users\Admin\Desktop\Test_Data\output.csv"
    
    # 3. Execute
    if os.path.exists(root) and "Your_split_pdfs" not in root:
        convert_all_output_folders(root, out_csv)
    elif "Your_split_pdfs" in root:
        print("❌ Path not configured, cannot run. Please start via pipeline or set path manually.")