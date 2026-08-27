import os
import pandas as pd
import re
import csv
from pathlib import Path

# ===== 配置路径 (Pipeline 将替换这些变量) =====
# 默认值仅用于单独测试
DICT_DIR = Path("dict")  
INPUT_CSV = Path(r"Data/Processed_Results/Standardized_Result.csv")
OUTPUT_CSV = Path(r"Data/Processed_Results/Final_Classified_Result.csv")

# ===== 目标列 =====
TARGET_COLS = [
    "Polymer A Name", "Polymer B Name", "Polymer C Name",
    "Polymer D Name", "Polymer E Name", "Material A Name",
    "Material B Name", "Material C Name", "Material D Name",
    "Material E Name", "ComponentName"
]

# ===== 匹配配置 =====
MATCH_WHOLE_CELL = True
NORMALIZE = True
SEP_RE = re.compile(r"[\s_\-\,\./()]+")

# ===== 全局变量 =====
EXACT_MAP = {}
SUBSTR_PATTERNS = []
KW_TO_CAT = {}

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = (s.replace("（", "(").replace("）", ")")
           .replace("，", ",").replace("／", "/")
           .replace("－", "-").replace("\u00A0", " "))
    return SEP_RE.sub("", s)

def read_dict_lines(p: Path):
    """读 txt：去行内#注释、空白、大小写去重。"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936", "latin1"):
        try:
            raw = p.read_text(encoding=enc, errors="ignore"); break
        except Exception:
            continue
    else:
        print(f"⚠️ 警告：无法读取字典文件 {p.name}")
        return []
        
    out, seen = [], set()
    for line in raw.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        line = line.strip()
        if not line:
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out

def load_dictionaries(dict_dir: Path):
    """
    遍历 dict_dir 下的 Material 和 Polymer 子文件夹读取字典
    """
    if not dict_dir.exists():
        raise FileNotFoundError(f"找不到字典根目录：{dict_dir}")

    kw_to_cat = {}
    sub_folders = ["Material", "Polymer"]
    found_files = 0
    
    for sub in sub_folders:
        target_dir = dict_dir / sub
        if not target_dir.exists():
            # 尝试不区分大小写查找文件夹 (兼容性)
            if target_dir.name.lower() in [x.name.lower() for x in dict_dir.iterdir() if x.is_dir()]:
                # 找到实际存在的对应文件夹
                target_dir = next(x for x in dict_dir.iterdir() if x.is_dir() and x.name.lower() == target_dir.name.lower())
            else:
                print(f"⚠️ 警告：字典子文件夹未找到：{target_dir}")
                continue
            
        for fp in sorted(target_dir.glob("*.txt")):
            found_files += 1
            cat = fp.stem.strip()
            if not cat: continue
            for kw in read_dict_lines(fp):
                kn = normalize(kw) if NORMALIZE else kw.lower()
                if not kn: continue
                if kn not in kw_to_cat or cat < kw_to_cat[kn]:
                    kw_to_cat[kn] = cat

    if found_files == 0:
        print(f"⚠️ 警告：在 {dict_dir} 中未发现任何 .txt 字典文件，将只进行原样输出。")

    exact_map = kw_to_cat
    all_kws = sorted(kw_to_cat.keys(), key=len, reverse=True)
    CHUNK_SIZE = 4000
    substr_patterns = []
    for i in range(0, len(all_kws), CHUNK_SIZE):
        chunk = all_kws[i:i + CHUNK_SIZE]
        # 转义并合并正则
        try:
            pattern = re.compile("|".join(re.escape(k) for k in chunk))
            substr_patterns.append(pattern)
        except Exception as e:
            print(f"⚠️ 正则编译错误 (跳过该块): {e}")

    return exact_map, substr_patterns, kw_to_cat

def get_match_text(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    return s if MATCH_WHOLE_CELL else s.split()[0]

def find_best_category(text: str):
    t = normalize(text) if NORMALIZE else text.lower()
    if not t:
        return None
    return EXACT_MAP.get(t, None)

def detect_encoding(path: Path):
    """只检测编码，不再推测分隔符"""
    with open(path, "rb") as fb:
        head = fb.read(8192)
    enc_guess = None
    if head.startswith(b"\xef\xbb\xbf"):
        enc_guess = "utf-8-sig"
    elif head.startswith(b"\xff\xfe"):
        enc_guess = "utf-16-le"
    elif head.startswith(b"\xfe\xff"):
        enc_guess = "utf-16-be"

    for enc in [enc_guess, "utf-8", "utf-8-sig", "gbk", "cp936", "latin1"]:
        if not enc: continue
        try:
            head.decode(enc, errors="strict")
            return enc
        except Exception:
            continue
    return "utf-8" # 默认兜底

def main():
    global EXACT_MAP, SUBSTR_PATTERNS, KW_TO_CAT
    
    d_dir = Path(DICT_DIR)
    in_csv = Path(INPUT_CSV)
    out_csv = Path(OUTPUT_CSV)

    print(f"🔧 字典目录: {d_dir}")
    print(f"📥 输入文件: {in_csv}")
    
    # 加载字典
    try:
        EXACT_MAP, SUBSTR_PATTERNS, KW_TO_CAT = load_dictionaries(d_dir)
    except Exception as e:
        print(f"❌ 字典加载失败: {e}")
        return

    if not in_csv.exists():
        raise FileNotFoundError(f"找不到输入 CSV：{in_csv}")

    # 1. 检测编码
    enc = detect_encoding(in_csv)
    # 2. 强制使用逗号作为分隔符
    sep = "," 
    print(f"✅ 读取编码={enc} 分隔符={repr(sep)} (强制)")

    try:
        # 用 pandas 读 CSV
        # on_bad_lines='skip': 跳过格式错误的行，防止程序直接崩溃
        df = pd.read_csv(in_csv, dtype=str, encoding=enc, sep=sep, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        # 尝试备用方案：如果强制逗号失败，尝试自动探测引擎
        print("🔄 尝试使用 python 引擎读取...")
        try:
            df = pd.read_csv(in_csv, dtype=str, encoding=enc, sep=None, engine='python', on_bad_lines='skip')
        except Exception as e2:
            print(f"❌ 依然失败: {e2}")
            return

    summary = {}
    for col in TARGET_COLS:
        if col not in df.columns:
            continue

        orig_vals = df[col].tolist()
        new_vals, replaced, skipped = [], 0, 0

        for orig in orig_vals:
            text = get_match_text(orig)
            if not text:
                new_vals.append(orig); skipped += 1; continue

            cat = find_best_category(text)
            if cat is not None:
                new_vals.append(cat); replaced += 1
            else:
                new_vals.append(orig)

        df[col] = new_vals
        summary[col] = {"total": len(orig_vals), "replaced": replaced, "skipped": skipped}

    if not out_csv.parent.exists():
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 输出 CSV (强制使用 utf-8-sig 和 逗号)
    try:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig", sep=",")
        print("=== 归类替换完成 ===")
        print(f"📤 输出文件：{out_csv}")
        for col, st in summary.items():
            print(f"  - {col}: 替换 {st['replaced']} / {st['total']}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")

if __name__ == "__main__":
    main()