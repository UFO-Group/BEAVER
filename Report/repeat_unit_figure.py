# Agent/Report/repeat_unit_figure.py
# -*- coding: utf-8 -*-
"""
Generate representative repeat-unit figures for Design-mode reports.

Dictionary-based version (v4: architecture-aware multi-segment extraction).

Workflow:
1. LLM extracts conservative polymer / repeat-unit / segment NAMES from the final report.
   The LLM is not allowed to generate SMILES.
2. A user-provided alias source maps extracted names / aliases to canonical names.
   This can be a txt keyword folder: each .txt filename is treated as the
   canonical/database name, and file contents are treated as aliases/keywords.
3. A user-provided name-SMILES CSV maps canonical names to SMILES.
4. RDKit validates the mapped SMILES and renders parseable structures.
5. A schematic fallback is generated when exact SMILES are unavailable or invalid.
6. A Markdown image block is inserted into the report.

"""

from __future__ import annotations

import csv
import json
import os
import re
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Callable
from difflib import SequenceMatcher

_AMBIGUOUS_ALIAS = "__AMBIGUOUS_ALIAS__"


# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except Exception:
    Chem = None
    Draw = None
    HAS_RDKIT = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    HAS_PIL = False


# -----------------------------------------------------------------------------
# DeepSeek wrapper
# -----------------------------------------------------------------------------
def _default_deepseek_callable() -> Callable[..., str] | None:
    """Import lazily so this module can still be imported in offline/unit-test mode."""
    try:
        from Agent.Agent_Config.deepseek_client import call_deepseek_llm  # type: ignore
        return call_deepseek_llm
    except Exception:
        return None


def _call_llm(llm_callable: Callable[..., str], prompt: str, system_prompt: str, temperature: float = 0.0) -> str:
    """Handle the slightly different signatures used by local DeepSeek wrappers."""
    try:
        return llm_callable(prompt=prompt, system_prompt=system_prompt, temperature=temperature)
    except TypeError:
        pass
    try:
        return llm_callable(prompt, system_prompt=system_prompt, temperature=temperature)
    except TypeError:
        pass
    try:
        return llm_callable(prompt, temperature=temperature)
    except TypeError:
        pass
    return llm_callable(prompt)


def _strip_think_tags(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Robustly extract one JSON object from an LLM response."""
    if not text:
        return None

    text = _strip_think_tags(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text

    if not fenced:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start:end + 1]

    repaired = re.sub(r"(?m)//.*$", "", candidate)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    for s in (candidate, repaired):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _sanitize_filename(text: str, max_length: int = 80) -> str:
    text = str(text or "repeat_unit").strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "repeat_unit")[:max_length]


# -----------------------------------------------------------------------------
# Name normalization, dictionary loading, and SMILES resolution
# -----------------------------------------------------------------------------
def _normalize_name_key(value: Any, *, compact: bool = False) -> str:
    """Normalize polymer names for robust exact/fuzzy lookup."""
    s = str(value or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    replacements = {
        "–": "-", "—": "-", "−": "-", "‑": "-",
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
        "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Ε": "epsilon",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    if compact:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return s


def _split_alias_cell(value: Any) -> list[str]:
    """Split multi-alias cells conservatively; avoid splitting chemical names on commas."""
    s = str(value or "").strip()
    if not s:
        return []
    parts = re.split(r"\s*(?:;|；|\||\n)\s*", s)
    return [p.strip() for p in parts if p and p.strip()]


def _split_keyword_text(value: Any) -> list[str]:
    """Split aliases from txt keyword files.

    Txt keyword files are looser than CSV dictionaries: usually one keyword per
    line, but Chinese/English separators are also supported. Comment lines
    beginning with # or // are ignored.
    """
    s = str(value or "").strip()
    if not s:
        return []
    cleaned_lines: list[str] = []
    for line in s.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        line = re.sub(r"\s+#.*$", "", line).strip()
        line = re.sub(r"\s+//.*$", "", line).strip()
        if line:
            cleaned_lines.append(line)
    if not cleaned_lines:
        return []
    merged = "\n".join(cleaned_lines)
    parts = re.split(r"\s*(?:;|；|\||\n|，|、)\s*", merged)

    expanded: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # In keyword txt files, English commas usually separate aliases.
        # To avoid damaging long IUPAC-like names, only split shorter cells.
        if "," in part and len(part) < 120:
            expanded.extend([x.strip() for x in part.split(",") if x.strip()])
        else:
            expanded.append(part)
    return [p for p in expanded if p]


def _read_text_with_fallback_encodings(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return p.read_text(encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read text file {p}: {last_err}")


def load_keyword_folder_dictionary(folder_path: str | os.PathLike[str]) -> dict[str, str]:
    """Load a txt keyword folder as alias -> canonical-name mapping.

    Rule:
      Polymer-关键词/PLA.txt
        contains: PLA; poly(lactic acid); polylactide; 聚乳酸

    Then every alias above maps to canonical name "PLA", because the txt
    filename stem is treated as the database-facing standard name.
    """
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Keyword folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected keyword folder, got file: {root}")

    alias_to_canonical: dict[str, str] = {}
    for txt_path in sorted(root.rglob("*.txt")):
        canonical = txt_path.stem.strip()
        if not canonical:
            continue
        aliases = [canonical]
        try:
            aliases.extend(_split_keyword_text(_read_text_with_fallback_encodings(txt_path)))
        except Exception:
            aliases = [canonical]

        aliases.extend([
            canonical.replace("_", " "),
            canonical.replace("-", " "),
            canonical.replace("_", "-"),
        ])

        for alias in aliases:
            alias = str(alias or "").strip()
            if not alias:
                continue
            for key in (_normalize_name_key(alias), _normalize_name_key(alias, compact=True)):
                if not key:
                    continue
                old_value = alias_to_canonical.get(key)
                if old_value and old_value != canonical:
                    # The same keyword appears in more than one txt file, for example
                    # "PGA" may mean poly(glycolic acid) or polyglutamic acid.
                    # Mark it ambiguous so the resolver requires a more specific name.
                    alias_to_canonical[key] = _AMBIGUOUS_ALIAS
                else:
                    alias_to_canonical[key] = canonical

    return {k: v for k, v in alias_to_canonical.items() if k}


def _candidate_name_variants(raw_name: str) -> list[str]:
    """Generate conservative lookup variants from an LLM-extracted name."""
    raw = str(raw_name or "").strip()
    if not raw or _is_generic_polymer_name(raw):
        return []
    variants: list[str] = [raw]

    variants.extend(re.findall(r"\(([^()]{2,80})\)", raw))

    # PLA-based copolymer -> PLA
    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_+./()\-]{1,60})\s*[- ]based\b", raw, flags=re.I)
    if m:
        variants.append(m.group(1))

    # PBS based rigid hydrophilic copolyester -> PBS
    m2 = re.match(r"^\s*([A-Za-z]{2,12})\s+based\b", raw, flags=re.I)
    if m2:
        variants.append(m2.group(1))

    # self-reinforced PLA composite -> PLA
    variants.extend(re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,12})(?![A-Za-z0-9])", raw))

    cleaned = re.sub(
        r"\b(random|block|graft|segmented|alternating|copolymer|polymer|polyester|composite|blend|network|based)\b",
        " ",
        raw,
        flags=re.I,
    )
    cleaned = re.sub(r"[-_/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned != raw:
        variants.append(cleaned)

    seen = set()
    out: list[str] = []
    for v in variants:
        v = str(v or "").strip()
        if not v:
            continue
        key = _normalize_name_key(v, compact=True)
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out




def _is_generic_polymer_name(name: Any) -> bool:
    """Return True for non-drawable class-level names such as polyester/polymer.

    This guard is intentionally conservative: specific abbreviations (PLA, PEG,
    PCL, PBS, PEF, PGA, etc.) and chemically specific names such as
    poly(ethylene glycol) are not treated as generic.
    """
    raw = str(name or "").strip()
    if not raw:
        return True

    # Specific all-caps abbreviations should be allowed.
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,12}", raw):
        return False

    norm = _normalize_name_key(raw)
    norm = re.sub(r"[()_\-/]+", " ", norm)
    tokens = re.findall(r"[a-z0-9]+", norm)
    if not tokens:
        return True

    generic_terms = {
        "polymer", "polymers", "polymeric",
        "polyester", "polyesters",
        "copolymer", "copolymers",
        "copolyester", "copolyesters",
        "matrix", "matrices", "backbone",
        "segment", "segments", "unit", "units", "repeat", "repeating",
        "material", "materials", "candidate", "design", "structure",
        "composite", "composites", "blend", "blends", "network", "networks",
        "thermoplastic", "elastomer", "resin",
    }
    modifiers = {
        "a", "an", "the", "and", "or", "of", "with", "for", "to", "in",
        "based", "containing", "reinforced", "modified", "proposed", "designed",
        "biodegradable", "degradable", "hydrolytic", "hydrolysable", "hydrolytically",
        "aliphatic", "aromatic", "semi", "semicrystalline", "crystalline", "amorphous",
        "rigid", "flexible", "soft", "hard", "high", "low", "thermal", "heat",
        "resistant", "hydrophilic", "hydrophobic", "ester", "rich", "phase", "separated",
        "main", "specific", "unspecified", "unknown", "general", "generic", "tbd",
        "tg", "tm", "glass", "transition", "temperature",
    }

    content_tokens = [t for t in tokens if t not in modifiers]
    if not content_tokens:
        return True

    # Class-level terms only -> generic. Any remaining chemically specific token -> not generic.
    return all(t in generic_terms for t in content_tokens)


def _filter_generic_candidates(candidates: list[str]) -> list[str]:
    """Remove generic polymer-class terms from lookup candidates."""
    return [c for c in candidates if not _is_generic_polymer_name(c)]


def _read_csv_like_rows(path: str | os.PathLike[str]) -> tuple[list[dict[str, str]], list[str]]:
    """Read CSV/TSV using common encodings; return rows and headers."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping file does not exist: {p}")

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(p, "r", encoding=enc, newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                except Exception:
                    dialect = csv.excel
                try:
                    has_header = csv.Sniffer().has_header(sample)
                except Exception:
                    has_header = True

                if has_header:
                    reader = csv.DictReader(f, dialect=dialect)
                    headers = [h or "" for h in (reader.fieldnames or [])]
                    rows = [{str(k or ""): str(v or "") for k, v in row.items()} for row in reader]
                    return rows, headers

                reader2 = csv.reader(f, dialect=dialect)
                raw_rows = list(reader2)
                if not raw_rows:
                    return [], []
                width = max(len(r) for r in raw_rows)
                headers = [f"col_{i}" for i in range(width)]
                rows = []
                for r in raw_rows:
                    rows.append({headers[i]: str(r[i] if i < len(r) else "") for i in range(width)})
                return rows, headers
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read mapping file {p}: {last_err}")


def _read_json_mapping(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(p, "r", encoding=enc) as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
            raise ValueError("JSON mapping must be an object/dict.")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read JSON mapping {p}: {last_err}")


def _header_key(header: str) -> str:
    return re.sub(r"[\s_\-()/\\]+", "", str(header or "").strip().lower())


def _guess_column(headers: list[str], candidates: list[str]) -> str | None:
    if not headers:
        return None
    hmap = {_header_key(h): h for h in headers}
    cand_keys = [_header_key(c) for c in candidates]

    for ck in cand_keys:
        if ck in hmap:
            return hmap[ck]
    for h in headers:
        hk = _header_key(h)
        if any(ck and ck in hk for ck in cand_keys):
            return h
    return None


def _looks_like_smiles(value: Any) -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    if re.search(r"\s", s) and not re.search(r"\[[^\]]+\]", s):
        return False
    return bool(re.search(r"[=#@\[\]\(\)\\/]|\bBr\b|\bCl\b|[CONSPFBIconspfb]", s))


def _guess_smiles_column_by_values(rows: list[dict[str, str]], headers: list[str]) -> str | None:
    if not rows or not headers:
        return None
    best_col = None
    best_score = -1
    sample = rows[:50]
    for h in headers:
        vals = [r.get(h, "") for r in sample]
        nonempty = [v for v in vals if str(v).strip()]
        if not nonempty:
            continue
        score = sum(1 for v in nonempty if _looks_like_smiles(v)) / max(1, len(nonempty))
        if score > best_score:
            best_score = score
            best_col = h
    return best_col if best_score >= 0.35 else None


def _best_fuzzy_match(query_key: str, choices: list[str], cutoff: float = 0.86) -> tuple[str | None, float]:
    if not query_key or not choices:
        return None, 0.0
    best_key = None
    best_score = 0.0
    for c in choices:
        score = SequenceMatcher(None, query_key, c).ratio()
        if score > best_score:
            best_key = c
            best_score = score
    if best_key is not None and best_score >= cutoff:
        return best_key, best_score
    return None, best_score


def load_name_name_dictionary(path: str | os.PathLike[str] | None) -> dict[str, str]:
    """
    Load alias/extracted-name -> canonical/database-name dictionary.

    Supported sources:
    1. txt keyword folder: each .txt filename stem is the canonical name; file
       contents are aliases/keywords. This is the intended mode for
       C:/Users/user/Desktop/关键词归类/Polymer-关键词.
    2. CSV/JSON mapping file: alias/source name -> canonical/database name.

    Returns a dict keyed by normalized names. Values preserve the canonical name text.
    If path is None or empty, returns an empty dict.
    """
    if not path:
        return {}
    p = Path(path)

    if p.is_dir():
        return load_keyword_folder_dictionary(p)

    alias_to_canonical: dict[str, str] = {}

    if p.suffix.lower() == ".json":
        obj = _read_json_mapping(p)
        for alias, canonical in obj.items():
            canonical_s = str(canonical or "").strip()
            if not canonical_s:
                continue
            for a in _split_alias_cell(alias):
                alias_to_canonical[_normalize_name_key(a)] = canonical_s
                alias_to_canonical[_normalize_name_key(a, compact=True)] = canonical_s
            alias_to_canonical[_normalize_name_key(canonical_s)] = canonical_s
            alias_to_canonical[_normalize_name_key(canonical_s, compact=True)] = canonical_s
        return {k: v for k, v in alias_to_canonical.items() if k}

    rows, headers = _read_csv_like_rows(p)
    if not rows:
        return {}

    alias_col = _guess_column(headers, [
        "alias", "synonym", "source_name", "raw_name", "input_name", "extracted_name",
        "llm_name", "report_name", "name_from_report", "original_name", "别名", "同义名",
        "原名称", "提取名称", "报告名称", "LLM名称",
    ])
    canonical_col = _guess_column(headers, [
        "canonical_name", "standard_name", "database_name", "target_name", "db_name",
        "normalized_name", "name", "polymer_name", "material_name", "标准名称", "规范名称",
        "数据库名称", "目标名称", "名称", "聚合物名称", "材料名称",
    ])

    if not alias_col or not canonical_col:
        usable_headers = [h for h in headers if h]
        if len(usable_headers) >= 2:
            alias_col, canonical_col = usable_headers[0], usable_headers[1]
        else:
            return {}

    for row in rows:
        canonical = str(row.get(canonical_col, "") or "").strip()
        alias_cell = str(row.get(alias_col, "") or "").strip()
        if not canonical or not alias_cell:
            continue
        for alias in _split_alias_cell(alias_cell):
            alias_to_canonical[_normalize_name_key(alias)] = canonical
            alias_to_canonical[_normalize_name_key(alias, compact=True)] = canonical
        alias_to_canonical[_normalize_name_key(canonical)] = canonical
        alias_to_canonical[_normalize_name_key(canonical, compact=True)] = canonical

    return {k: v for k, v in alias_to_canonical.items() if k}


def load_name_smiles_dictionary(path: str | os.PathLike[str] | None) -> dict[str, dict[str, str]]:
    """
    Load canonical/database-name -> SMILES mapping from CSV/TSV.

    Returns a dict keyed by normalized names. Each value contains:
      {"name": original_name, "smiles": smiles}
    """
    if not path:
        return {}
    p = Path(path)
    rows, headers = _read_csv_like_rows(p)
    if not rows:
        return {}

    smiles_col = _guess_column(headers, [
        "smiles", "smile", "canonical_smiles", "canonical_smile", "isomeric_smiles",
        "SMILES", "SMILE", "结构", "结构式", "smiles结构",
    ]) or _guess_smiles_column_by_values(rows, headers)

    name_col = _guess_column(headers, [
        "canonical_name", "standard_name", "database_name", "name", "polymer_name",
        "material_name", "polymer", "material", "名称", "标准名称", "规范名称",
        "数据库名称", "聚合物名称", "材料名称", "Polymer A", "Polymer", "Material",
    ])

    if not smiles_col:
        usable_headers = [h for h in headers if h]
        if len(usable_headers) >= 2:
            smiles_col = usable_headers[1]
        else:
            raise ValueError(f"Could not detect SMILES column in {p}")

    if not name_col or name_col == smiles_col:
        usable_headers = [h for h in headers if h and h != smiles_col]
        if usable_headers:
            name_col = usable_headers[0]
        else:
            raise ValueError(f"Could not detect name column in {p}")

    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get(name_col, "") or "").strip()
        smiles = str(row.get(smiles_col, "") or "").strip()
        if not name or not smiles:
            continue
        record = {"name": name, "smiles": smiles}
        mapping[_normalize_name_key(name)] = record
        mapping[_normalize_name_key(name, compact=True)] = record
    return {k: v for k, v in mapping.items() if k}


def validate_smiles_with_rdkit(smiles: str) -> dict[str, Any]:
    """Validate SMILES with RDKit and return a compact validation record."""
    s = str(smiles or "").strip()
    result: dict[str, Any] = {
        "input_smiles": s,
        "rdkit_available": HAS_RDKIT,
        "valid": False,
        "canonical_smiles": "",
        "error": "",
    }
    if not s:
        result["error"] = "empty_smiles"
        return result
    if not HAS_RDKIT:
        result["error"] = "rdkit_unavailable"
        return result
    try:
        mol = Chem.MolFromSmiles(s)  # type: ignore[union-attr]
        if mol is None:
            result["error"] = "Chem.MolFromSmiles returned None"
            return result
        try:
            Chem.SanitizeMol(mol)  # type: ignore[union-attr]
        except Exception as sanitize_err:
            result["error"] = f"sanitize_failed: {sanitize_err}"
            return result
        result["valid"] = True
        result["canonical_smiles"] = Chem.MolToSmiles(mol)  # type: ignore[union-attr]
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def _lookup_alias_to_canonical(
    raw_name: str,
    alias_map: dict[str, str],
    *,
    allow_fuzzy: bool = True,
    fuzzy_cutoff: float = 0.86,
) -> tuple[str, dict[str, Any]]:
    """Resolve extracted/raw name to canonical/database name using the alias dictionary."""
    raw = str(raw_name or "").strip()
    if not raw:
        return "", {"status": "empty_name", "match_method": "none"}

    variants = _candidate_name_variants(raw) or [raw]

    ambiguous_hits: list[dict[str, Any]] = []
    for variant in variants:
        for compact in (False, True):
            key = _normalize_name_key(variant, compact=compact)
            if key and key in alias_map:
                mapped = alias_map[key]
                if mapped == _AMBIGUOUS_ALIAS:
                    ambiguous_hits.append({
                        "matched_key": key,
                        "matched_variant": variant,
                        "match_method": "alias_exact_compact" if compact else "alias_exact",
                    })
                    continue
                return mapped, {
                    "status": "matched",
                    "match_method": "alias_exact_compact" if compact else "alias_exact",
                    "matched_key": key,
                    "matched_variant": variant,
                    "score": 1.0,
                }

    if allow_fuzzy and alias_map:
        compact_choices = [k for k, v in alias_map.items() if k and v != _AMBIGUOUS_ALIAS]
        best_match = None
        best_score = 0.0
        best_variant = ""
        for variant in variants:
            compact_key = _normalize_name_key(variant, compact=True)
            matched_key, score = _best_fuzzy_match(compact_key, compact_choices, cutoff=fuzzy_cutoff)
            if matched_key and score > best_score:
                best_match = matched_key
                best_score = score
                best_variant = variant
        if best_match:
            return alias_map[best_match], {
                "status": "matched",
                "match_method": "alias_fuzzy",
                "matched_key": best_match,
                "matched_variant": best_variant,
                "score": round(best_score, 4),
            }

    if ambiguous_hits:
        return raw, {
            "status": "ambiguous_alias_ignored_used_raw_name",
            "match_method": "raw_name",
            "score": 0.0,
            "ambiguous_hits": ambiguous_hits[:6],
        }

    return raw, {"status": "unmapped_alias_used_raw_name", "match_method": "raw_name", "score": 0.0}



def _lookup_alias_to_canonical_strict_exact(
    raw_name: str,
    alias_map: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """Resolve a FULL polymer name by strict exact alias lookup only.

    This is used only in the overall-polymer priority stage.
    It deliberately does NOT expand names such as PBS-b-PEG into PBS / PEG.
    Therefore, PBS-b-PEG can match only PBS-b-PEG or its exact compact form,
    not the component aliases PBS or PEG.
    """
    raw = str(raw_name or "").strip()
    if not raw:
        return "", {"status": "empty_name", "match_method": "none"}

    keys = [
        _normalize_name_key(raw),
        _normalize_name_key(raw, compact=True),
    ]

    for key in keys:
        if key and key in alias_map:
            mapped = alias_map[key]
            if mapped == _AMBIGUOUS_ALIAS:
                return raw, {
                    "status": "ambiguous_full_alias",
                    "match_method": "strict_alias_exact",
                    "matched_key": key,
                    "matched_variant": raw,
                    "score": 0.0,
                }
            return mapped, {
                "status": "matched",
                "match_method": "strict_alias_exact",
                "matched_key": key,
                "matched_variant": raw,
                "score": 1.0,
            }

    return raw, {
        "status": "full_alias_not_found",
        "match_method": "strict_alias_exact",
        "matched_variant": raw,
        "score": 0.0,
    }

def _lookup_canonical_to_smiles(
    canonical_name: str,
    smiles_map: dict[str, dict[str, str]],
    *,
    allow_fuzzy: bool = True,
    fuzzy_cutoff: float = 0.86,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Resolve canonical/database name to SMILES using the name-SMILES CSV."""
    canonical = str(canonical_name or "").strip()
    if not canonical:
        return None, {"status": "empty_canonical_name", "match_method": "none"}

    for compact in (False, True):
        key = _normalize_name_key(canonical, compact=compact)
        if key and key in smiles_map:
            return smiles_map[key], {
                "status": "matched",
                "match_method": "smiles_exact_compact" if compact else "smiles_exact",
                "matched_key": key,
                "score": 1.0,
            }

    if allow_fuzzy and smiles_map:
        compact_key = _normalize_name_key(canonical, compact=True)
        choices = [k for k in smiles_map.keys() if k]
        matched_key, score = _best_fuzzy_match(compact_key, choices, cutoff=fuzzy_cutoff)
        if matched_key:
            return smiles_map[matched_key], {
                "status": "matched",
                "match_method": "smiles_fuzzy",
                "matched_key": matched_key,
                "score": round(score, 4),
            }

    return None, {"status": "smiles_not_found", "match_method": "none", "score": 0.0}




def _split_polymer_combo_name(polymer_name: Any) -> list[str]:
    """Split names such as PEF-b-PGA, PLA-co-PCL, PBS/PCL into lookup candidates."""
    raw = str(polymer_name or "").strip()
    if not raw:
        return []

    candidates: list[str] = [raw]

    # Capture abbreviations in parentheses: poly(ethylene furanoate) (PEF)
    candidates.extend(re.findall(r"\(([^()]{2,40})\)", raw))

    # Common copolymer / blend separators. Keep the original first.
    split_text = re.sub(
        r"(?i)\b(?:block|random|graft|alternating|segmented)\b",
        " ",
        raw,
    )
    split_text = re.sub(r"(?i)\s*-\s*(?:b|block|co|ran|g|alt)\s*-\s*", "|", split_text)
    split_text = re.sub(r"\s*(?:/|\\|\+|,|;|；|，|、|\band\b|\bwith\b)\s*", "|", split_text, flags=re.I)
    for part in split_text.split("|"):
        part = part.strip(" -_()[]{}")
        if part:
            candidates.append(part)

    # Extract standalone uppercase abbreviations from the full polymer name.
    candidates.extend(re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,12})(?![A-Za-z0-9])", raw))

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = str(c or "").strip()
        key = _normalize_name_key(c, compact=True)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _is_short_ambiguous_label(label: str) -> bool:
    """Return True for labels such as EF/GA that are too short to identify a polymer."""
    lab = str(label or "").strip()
    if not lab:
        return True
    compact = _normalize_name_key(lab, compact=True)
    return len(compact) <= 2


def _split_polymer_components_only(polymer_name: Any) -> list[str]:
    """Return only component tokens from names such as PEF-b-PGA, not the full combo."""
    raw = str(polymer_name or "").strip()
    if not raw:
        return []

    split_text = re.sub(r"(?i)\b(?:block|random|graft|alternating|segmented)\b", " ", raw)
    split_text = re.sub(r"(?i)\s*-\s*(?:b|block|co|ran|g|alt)\s*-\s*", "|", split_text)
    split_text = re.sub(r"\s*(?:/|\\|\+|,|;|；|，|、|\band\b|\bwith\b)\s*", "|", split_text, flags=re.I)

    comps: list[str] = []
    for part in split_text.split("|"):
        part = part.strip(" -_()[]{}")
        if part and part != raw:
            comps.append(part)

    if not comps:
        comps.extend(re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{2,12})(?![A-Za-z0-9])", raw))

    seen: set[str] = set()
    out: list[str] = []
    for c in comps:
        key = _normalize_name_key(c, compact=True)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _unit_index_by_identity(unit: dict[str, Any], units: list[dict[str, Any]]) -> int:
    for i, u in enumerate(units):
        if u is unit:
            return i
    try:
        return units.index(unit)
    except Exception:
        return -1


def _collect_lookup_candidates_for_unit(unit: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Collect dictionary lookup candidates for one LLM-extracted unit.

    v3 safety rule:
    - First try the chemically informative extracted_name.
    - Do not let very short labels such as EF or GA drive matching.
    - Do not use the full copolymer name such as PEF-b-PGA as a candidate for
      every repeat unit.
    - If polymer_name can be split and the number of components matches the
      number of repeat_units, use the component at the same index as a fallback.
    """
    raw_name = str(
        unit.get("extracted_name")
        or unit.get("structure_name")
        or unit.get("name")
        or ""
    ).strip()
    label = str(unit.get("label") or "").strip()
    polymer_name = str(spec.get("polymer_name") or spec.get("design_name") or "").strip()

    repeat_units = [u for u in (spec.get("repeat_units") or []) if isinstance(u, dict)]
    components = _split_polymer_components_only(polymer_name)
    idx = _unit_index_by_identity(unit, repeat_units)

    candidates: list[str] = []

    if raw_name:
        candidates.extend(_candidate_name_variants(raw_name))
        candidates.append(raw_name)

    if idx >= 0 and len(components) == len(repeat_units) and idx < len(components):
        candidates.append(components[idx])

    if label and not _is_short_ambiguous_label(label):
        candidates.extend(_candidate_name_variants(label))
        candidates.append(label)

    if len(repeat_units) <= 1 and polymer_name:
        candidates.extend(_candidate_name_variants(polymer_name))

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = str(c or "").strip()
        key = _normalize_name_key(c, compact=True)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _resolve_candidate_to_record(
    candidate: str,
    alias_map: dict[str, str],
    smiles_map: dict[str, dict[str, str]],
    *,
    allow_fuzzy: bool,
    fuzzy_cutoff: float,
) -> tuple[str, dict[str, str] | None, dict[str, Any], dict[str, Any]]:
    """Resolve one candidate name through alias mapping and then SMILES lookup."""
    if _is_generic_polymer_name(candidate):
        return str(candidate or ""), None, {
            "status": "skipped_generic_name",
            "match_method": "none",
            "score": 0.0,
        }, {
            "status": "skipped_before_smiles_lookup",
            "match_method": "none",
            "score": 0.0,
        }

    canonical, alias_status = _lookup_alias_to_canonical(
        candidate,
        alias_map,
        allow_fuzzy=allow_fuzzy,
        fuzzy_cutoff=fuzzy_cutoff,
    )

    # User-intended rule: extracted_name must first map through the keyword
    # folder / alias dictionary. Do not bypass it by fuzzy-searching the SMILES
    # CSV directly with a raw LLM name.
    if alias_map and alias_status.get("status") != "matched":
        return canonical, None, alias_status, {
            "status": "skipped_no_keyword_folder_match",
            "match_method": "none",
            "score": 0.0,
        }

    smiles_record, smiles_status = _lookup_canonical_to_smiles(
        canonical,
        smiles_map,
        allow_fuzzy=allow_fuzzy,
        fuzzy_cutoff=fuzzy_cutoff,
    )
    return canonical, smiles_record, alias_status, smiles_status


# -----------------------------------------------------------------------------
# Architecture-aware repeat-unit policy
# -----------------------------------------------------------------------------
def _normalize_architecture_label(architecture: Any) -> str:
    """Normalize free-form architecture labels into a small controlled vocabulary."""
    arch = _normalize_name_key(architecture)
    arch = arch.replace("_", "-")
    arch = re.sub(r"\s+", " ", arch).strip()

    if not arch:
        return "unknown"
    if "block" in arch:
        return "block copolymer"
    if "graft" in arch:
        return "graft copolymer"
    if "random" in arch or "statistical" in arch or "ran" in arch:
        return "random copolymer"
    if "segmented" in arch or "multiblock" in arch or "multi block" in arch:
        return "segmented copolymer"
    if "blend" in arch or "composite" in arch:
        return "blend/composite"
    if "network" in arch or "crosslink" in arch:
        return "network"
    if "copolymer" in arch or "copolyester" in arch or arch == "co polymer":
        return "random copolymer"
    if "homo" in arch:
        return "homopolymer"
    return arch or "unknown"


def _architecture_requires_multiple_units(spec: dict[str, Any]) -> bool:
    """Return True when the design architecture should normally be drawn with 2+ polymer units.

    This prevents copolymers / blocks / blends from being collapsed into a single
    polymer-level RDKit structure by polymer_level_priority.
    """
    arch = _normalize_architecture_label(spec.get("architecture"))
    polymer_name = str(spec.get("polymer_name") or spec.get("design_name") or "").strip().lower()
    composition_note = str(spec.get("composition_note") or "").strip().lower()
    text = " ".join([arch, polymer_name, composition_note])

    multi_arch = {
        "random copolymer",
        "block copolymer",
        "graft copolymer",
        "segmented copolymer",
        "blend/composite",
    }
    if arch in multi_arch:
        return True

    markers = [
        "-co-", " co ", " copolymer", "copolyester",
        "-b-", " block ", " diblock", " triblock", " multiblock",
        "-g-", " graft ",
        " blend", " composite", " / ", "/", " + ",
        "pbs:peg", "pbs/peg", "pef:pga", "pla:pcl",
        "modified", "pegylated",
    ]
    return any(m in text for m in markers)



def _architecture_is_explicit_independent_multi_component(spec: dict[str, Any]) -> bool:
    """Return True when the material is an explicit multi-polymer mixture/component system.

    These systems should remain multi-unit even if an overall label happens to
    match the dictionary, because their design meaning is separate components
    rather than one covalently named copolymer.
    """
    arch = _normalize_architecture_label(spec.get("architecture"))
    polymer_name = str(spec.get("polymer_name") or spec.get("design_name") or "").strip().lower()
    composition_note = str(spec.get("composition_note") or "").strip().lower()
    text = " ".join([arch, polymer_name, composition_note])

    if arch == "blend/composite":
        return True

    independent_markers = [
        " blend", "blended", " mixture", " physical mixture", " composite",
        "matrix", "polymer/polymer", "polymer blend", "co-blend",
    ]
    if any(m in text for m in independent_markers):
        return True

    # Slash/plus names without covalent copolymer markers are usually blends or
    # independent components, e.g. PLA/PCL or PBS + PEG. Do not classify
    # poly(A-co-B), A-b-B, or A-g-B as independent merely because they contain
    # separators.
    covalent_markers = ["-co-", " co ", "copolymer", "copolyester", "-b-", " block ", "-g-", " graft "]
    if ("/" in polymer_name or "+" in polymer_name) and not any(m in text for m in covalent_markers):
        return True

    return False


def _try_resolve_overall_polymer_unit(
    *,
    polymer_name: str,
    alias_map: dict[str, str],
    smiles_map: dict[str, dict[str, str]],
    allow_fuzzy: bool,
    fuzzy_cutoff: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Strictly resolve the full overall polymer_name before component fallback.

    Ordered rule:
    1. Try the complete polymer_name exactly, e.g. PBS-b-PEG or PBAT.
    2. Do NOT split or expand the name in this stage.
    3. If the complete name is not found, return None and let the caller split
       names containing -co-, -b-, -g-, /, +, etc. into component polymers.
    """
    polymer_name = str(polymer_name or "").strip()
    if not polymer_name or _is_generic_polymer_name(polymer_name):
        return None, []

    trials: list[dict[str, Any]] = []

    if alias_map:
        canonical, alias_status = _lookup_alias_to_canonical_strict_exact(
            polymer_name,
            alias_map,
        )
        if alias_status.get("status") != "matched":
            trial = {
                "candidate": polymer_name,
                "canonical_name": canonical,
                "database_name": canonical,
                "mapped_smiles": "",
                "rdkit_valid": False,
                "alias_status": alias_status,
                "smiles_status": {
                    "status": "skipped_no_strict_full_name_alias_match",
                    "match_method": "none",
                    "score": 0.0,
                },
                "rdkit_validation": validate_smiles_with_rdkit(""),
                "strict_full_name_match": True,
            }
            trials.append(trial)
            return None, trials
    else:
        # Without an alias map, only try exact direct lookup against the SMILES map.
        canonical = polymer_name
        alias_status = {
            "status": "alias_map_unavailable_used_raw_full_name",
            "match_method": "raw_full_name",
            "matched_variant": polymer_name,
            "score": 1.0,
        }

    smiles_record, smiles_status = _lookup_canonical_to_smiles(
        canonical,
        smiles_map,
        allow_fuzzy=False,
        fuzzy_cutoff=fuzzy_cutoff,
    )

    mapped_smiles = str((smiles_record or {}).get("smiles", "") or "").strip()
    db_name = str((smiles_record or {}).get("name", "") or "").strip()
    validation = validate_smiles_with_rdkit(mapped_smiles)
    rdkit_valid = bool(validation.get("valid"))

    trial = {
        "candidate": polymer_name,
        "canonical_name": canonical,
        "database_name": db_name or canonical,
        "mapped_smiles": mapped_smiles,
        "rdkit_valid": rdkit_valid,
        "alias_status": alias_status,
        "smiles_status": smiles_status,
        "rdkit_validation": validation,
        "strict_full_name_match": True,
    }
    trials.append(trial)

    if rdkit_valid:
        unit = {
            "extracted_name": polymer_name,
            "label": polymer_name,
            "role": "overall polymer structure",
            "lookup_candidates": [polymer_name],
            "selected_lookup_candidate": polymer_name,
            "canonical_name": canonical,
            "database_name": db_name or canonical,
            "mapped_smiles": mapped_smiles,
            "smiles": mapped_smiles,
            "rdkit_valid": True,
            "rdkit_validation": validation,
            "name_resolution": {
                "alias_status": alias_status,
                "smiles_status": smiles_status,
                "candidate_trials": trials[:12],
            },
        }
        return unit, trials

    return None, trials


def _has_copolymer_or_component_separator(name: Any) -> bool:

    s = str(name or "").strip()
    if not s:
        return False

    # 避免把 PLA-based cyclic lactone terpolymer 这种描述性名称误拆成 PLA + ...
    if re.search(r"(?i)\b[A-Z0-9]+-based\b", s):
        return False
    if re.search(r"(?i)\bbased\s+", s):
        return False

    return bool(re.search(
        r"(?i)("
        r"-co-|-b-|-block-|-g-|-graft-|-ran-|-alt-|"
        r"/|\+|:|"
        r"poly\([^)]*\)\s*-\s*poly\([^)]*\)|"
        r"^[A-Z0-9]{2,12}\s*-\s*[A-Z0-9]{2,12}$"
        r")",
        s,
    ))


def _split_full_polymer_name_after_overall_fail(
    polymer_name: Any,
    architecture: Any = "unknown",
) -> list[dict[str, Any]]:
    """Split a full copolymer/block/blend name only after full-name lookup fails.

    Examples:
      PBS-b-PEG -> PBS + PEG
      poly(butylene adipate-co-terephthalate)
        -> poly(butylene adipate) + poly(butylene terephthalate)

    This function never generates SMILES. It only prepares polymer-level names
    for subsequent dictionary lookup.
    """
    raw = str(polymer_name or "").strip()
    if not raw:
        return []

    arch = _normalize_architecture_label(architecture)
    candidates: list[str] = []

    # Case 0: poly(A)-poly(B), e.g. poly(ethylene glycol)-poly(lactic acid)
    poly_parts = re.findall(r"(?i)poly\(([^()]+)\)", raw)
    if len(poly_parts) >= 2:
        candidates.extend([_wrap_component_as_polymer_name(x) for x in poly_parts])

    # Case 1: poly(A-co-B), poly(A-b-B), poly(A-g-B), etc.
    if len(candidates) < 2:
        m = re.match(r"(?i)^\s*poly\((.+)\)\s*$", raw)
        if m:
            inner = m.group(1).strip()
            if re.search(r"(?i)\s*-\s*(?:co|b|block|g|graft|ran|alt)\s*-\s*", inner):
                parts = re.split(
                    r"(?i)\s*-\s*(?:co|b|block|g|graft|ran|alt)\s*-\s*",
                    inner,
                )
                parts = [p.strip() for p in parts if p.strip()]

                # Repair shared-prefix copolyester names:
                # butylene adipate-co-terephthalate -> butylene adipate + butylene terephthalate
                if len(parts) >= 2:
                    m_prefix = re.match(r"(?i)^(butylene|ethylene|propylene|hexamethylene)\s+(.+)$", parts[0])
                    if m_prefix:
                        shared_prefix = m_prefix.group(1)
                        repaired_parts = [parts[0]]
                        for p in parts[1:]:
                            if not re.match(r"(?i)^(butylene|ethylene|propylene|hexamethylene)\s+", p):
                                repaired_parts.append(f"{shared_prefix} {p}")
                            else:
                                repaired_parts.append(p)
                        parts = repaired_parts

                candidates.extend([_wrap_component_as_polymer_name(p) for p in parts])

    # Case 2: PBS-b-PEG, PEF-b-PGA, PLA/PCL, PBS + PEG, PBS:PEG.
    if len(candidates) < 2:
        split_text = re.sub(
            r"(?i)\s*-\s*(?:co|b|block|g|graft|ran|alt)\s*-\s*",
            "|",
            raw,
        )
        split_text = re.sub(r"\s*(?:/|\+|:)\s*", "|", split_text)

        # Special case: PEG-PLA, PBS-PCL, PLA-PCL
        if "|" not in split_text and re.fullmatch(r"\s*[A-Z0-9]{2,12}\s*-\s*[A-Z0-9]{2,12}\s*", raw):
            split_text = re.sub(r"\s*-\s*", "|", raw)

        for part in split_text.split("|"):
            part = part.strip(" -_()[]{}")
            if part and part != raw:
                candidates.append(_wrap_component_as_polymer_name(part))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for cand in candidates:
        cand = str(cand or "").strip()
        if not cand or _is_generic_polymer_name(cand):
            continue

        key = _normalize_name_key(cand, compact=True)
        if not key or key in seen:
            continue

        seen.add(key)
        label = _component_label_from_name(cand)
        out.append({
            "extracted_name": cand,
            "label": label,
            "role": _component_role_from_architecture(arch, len(out), label),
            "inferred_from_failed_overall_name": raw,
        })

    return out

def _architecture_is_homopolymer(spec: dict[str, Any]) -> bool:
    arch = _normalize_architecture_label(spec.get("architecture"))
    if arch == "homopolymer":
        return True
    if arch in {"random copolymer", "block copolymer", "graft copolymer", "segmented copolymer", "blend/composite", "network"}:
        return False
    return False


def _component_role_from_architecture(architecture: Any, idx: int, label: str = "") -> str:
    arch = _normalize_architecture_label(architecture)
    label_l = str(label or "").lower()
    if arch == "block copolymer":
        if "peg" in label_l:
            return "hydrophilic soft block"
        if any(x in label_l for x in ["pbs", "pla", "pcl", "pef", "pga"]):
            return "biodegradable polyester block"
        return "block copolymer segment"
    if arch in {"random copolymer", "segmented copolymer"}:
        return "copolymer segment"
    if arch == "graft copolymer":
        return "backbone segment" if idx == 0 else "grafted polymer segment"
    if arch == "blend/composite":
        return "polymer matrix/component" if idx == 0 else "secondary polymer component"
    return "polymeric segment"


def _component_label_from_name(name: str) -> str:
    """Create a short English label for a polymeric component."""
    raw = str(name or "").strip()
    if not raw:
        return "Segment"

    # Prefer explicit abbreviations such as PBS, PEG, PLA in the name.
    caps = re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,12})(?![A-Za-z0-9])", raw)
    if caps:
        return f"{caps[0]} segment"

    known = [
        ("poly(butylene succinate)", "PBS segment"),
        ("butylene succinate", "PBS segment"),
        ("poly(ethylene glycol)", "PEG segment"),
        ("ethylene glycol", "PEG segment"),
        ("polylactide", "PLA segment"),
        ("poly(lactic acid)", "PLA segment"),
        ("polycaprolactone", "PCL segment"),
        ("poly(ε-caprolactone)", "PCL segment"),
        ("poly(glycolic acid)", "PGA segment"),
        ("poly(ethylene furanoate)", "PEF segment"),
        ("isosorbide succinate", "Isosorbide succinate segment"),
    ]
    low = raw.lower()
    for key, label in known:
        if key.lower() in low:
            return label

    cleaned = re.sub(r"^poly\((.*?)\)$", r"\1", raw, flags=re.I).strip()
    cleaned = cleaned[:38]
    return f"{cleaned} segment" if cleaned else "Segment"


def _wrap_component_as_polymer_name(component: str) -> str:
    """Convert a named segment into a dictionary-matchable polymer name when possible."""
    c = str(component or "").strip(" -_()[]{}")
    if not c:
        return ""
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,12}", c):
        return c
    if re.match(r"(?i)^poly\s*\(", c) or re.match(r"(?i)^poly[a-z]", c):
        return c
    # Common inner names in poly(A-co-B) should become poly(A), poly(B).
    return f"poly({c})"


def _is_polymer_level_name(name: Any) -> bool:
    """Return True when a name is likely a polymer/database-level name.

    This deliberately treats short polymer abbreviations such as PBS, PEG, PBA,
    PBT, PLA, PCL, PEF and PGA as polymer-level names. It also treats names
    beginning with poly(...) or common contracted polymer names as polymer-level.
    Names such as "butylene adipate" or "butylene terephthalate" are repeat-unit
    fragments and should be converted to poly(...), not used directly.
    """
    raw = str(name or "").strip()
    if not raw:
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,12}", raw):
        return True
    low = raw.lower().strip()
    if re.match(r"^poly\s*\(", low):
        return True
    if re.match(r"^poly[a-z]", low):
        return True
    return False


def _repair_to_polymer_level_name(name: Any) -> str:
    """Repair LLM-extracted repeat-unit fragments into polymer-level names.

    Example:
      butylene adipate        -> poly(butylene adipate)
      butylene terephthalate -> poly(butylene terephthalate)

    This function never generates SMILES. It only makes the lookup name compatible
    with a polymer-name keyword dictionary / polymer-SMILES CSV.
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    if _is_polymer_level_name(raw):
        return raw
    if _is_generic_polymer_name(raw):
        return raw

    # Do not wrap obvious inorganic fillers / small additives. Those should be
    # moved to non_polymer_components by the architecture postprocessor.
    low = raw.lower()
    non_polymer_markers = [
        "hydroxyapatite", "hap", "calcium phosphate", "silica", "graphene",
        "nanocellulose", "catalyst", "solvent", "buffer", "nanoparticle",
    ]
    if any(m in low for m in non_polymer_markers):
        return raw

    # Chemically named copolymer segments must still be dictionary-matchable
    # polymer names. This is the key guard requested by the user: do not lookup
    # fragments such as "butylene adipate" directly.
    if re.search(
        r"(?i)(succinate|adipate|terephthalate|sebacate|carbonate|lactide|glycolide|caprolactone|ethylene glycol|butylene|furanoate|isosorbide)",
        raw,
    ):
        return _wrap_component_as_polymer_name(raw)

    return raw


def _repair_units_to_polymer_level_names(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every drawable repeat_units entry uses a polymer-level lookup name."""
    repaired: list[dict[str, Any]] = []
    for unit in units:
        u = dict(unit)
        old_name = str(u.get("extracted_name") or u.get("structure_name") or u.get("name") or u.get("label") or "").strip()
        new_name = _repair_to_polymer_level_name(old_name)
        if new_name and new_name != old_name:
            u["original_extracted_name"] = old_name
            u["extracted_name"] = new_name
            u["polymer_level_name_repair_used"] = True
            # Preserve short labels such as "Butylene adipate" only if no better
            # label exists; labels are for display, extracted_name is for lookup.
            if not str(u.get("label") or "").strip() or str(u.get("label")).strip() == old_name:
                u["label"] = _component_label_from_name(new_name)
        repaired.append(u)
    return repaired


def _infer_repeat_units_from_polymer_name(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer polymeric component units from an overall copolymer / blend name.

    This is a safety net when the LLM extracts only an overall name such as
    PBS-b-PEG or poly(butylene succinate-co-isosorbide succinate). It does not
    generate SMILES; it only creates dictionary-matchable names for later lookup.
    """
    raw = str(spec.get("polymer_name") or spec.get("design_name") or "").strip()
    if not raw:
        return []

    architecture = spec.get("architecture", "unknown")
    candidates: list[str] = []

    # Case 1: poly(A-co-B), poly(A-b-B), poly(A-g-B), etc.
    m = re.search(r"(?i)^\s*poly\((.+)\)\s*$", raw)
    if m:
        inner = m.group(1).strip()
        if re.search(r"(?i)\s*-\s*(?:co|b|block|g|graft|ran|alt)\s*-\s*", inner):
            parts = re.split(r"(?i)\s*-\s*(?:co|b|block|g|graft|ran|alt)\s*-\s*", inner)
            candidates.extend([_wrap_component_as_polymer_name(x) for x in parts if x.strip()])

    # Case 2: poly(A)-b-poly(B), PBS-b-PEG, PBS/PEG, PLA + PCL, etc.
    if len(candidates) < 2:
        candidates.extend(_split_polymer_components_only(raw))

    # Case 3: explicit abbreviations around separators.
    if len(candidates) < 2 and re.search(r"(?:-|/|\+|:)" , raw):
        abbrev_parts = re.split(r"(?i)\s*(?:-\s*(?:co|b|block|g|graft|ran|alt)\s*-|/|\+|:)\s*", raw)
        candidates.extend([x.strip() for x in abbrev_parts if x.strip()])

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in candidates:
        name = _wrap_component_as_polymer_name(c)
        if not name or _is_generic_polymer_name(name):
            continue
        key = _normalize_name_key(name, compact=True)
        if not key or key in seen:
            continue
        seen.add(key)
        label = _component_label_from_name(name)
        out.append({
            "extracted_name": name,
            "label": label,
            "role": _component_role_from_architecture(architecture, len(out), label),
            "inferred_from_polymer_name": raw,
        })

    return out


def _deduplicate_repeat_units(units: list[Any]) -> list[dict[str, Any]]:
    """Deduplicate repeat_units by normalized extracted name / label."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in units:
        if not isinstance(unit, dict):
            unit = {"extracted_name": str(unit), "label": str(unit), "role": ""}
        else:
            unit = dict(unit)
        name = str(unit.get("extracted_name") or unit.get("structure_name") or unit.get("name") or unit.get("label") or "").strip()
        label = str(unit.get("label") or name).strip()
        key = _normalize_name_key(name or label, compact=True)
        if not key or key in seen:
            continue
        seen.add(key)
        unit.setdefault("extracted_name", name)
        unit.setdefault("label", label or name or "repeat unit")
        unit.setdefault("role", "")
        out.append(unit)
    return out


def _postprocess_repeat_units_by_architecture(spec: dict[str, Any]) -> dict[str, Any]:
    """Make repeat_units consistent with architecture.

    Key policy:
    - homopolymer: usually one drawable polymer structure;
    - copolymer/block/graft/segmented/blend/composite: normally 2+ polymeric units;
    - non-polymer fillers such as HAp stay in non_polymer_components rather than repeat_units.
    """
    spec = dict(spec or {})
    arch = _normalize_architecture_label(spec.get("architecture"))
    spec["architecture"] = arch

    units = spec.get("repeat_units") or []
    if not isinstance(units, list):
        units = []
    units = _deduplicate_repeat_units(units)

    requires_multi = _architecture_requires_multiple_units(spec)

    # Even for copolymers/block copolymers, repeat_units must remain
    # polymer-level lookup names. For example, use poly(butylene adipate)
    # rather than the fragment name butylene adipate.
    if requires_multi:
        units = _repair_units_to_polymer_level_names(units)
        units = _deduplicate_repeat_units(units)

    if requires_multi and len(units) < 2:
        inferred = _infer_repeat_units_from_polymer_name(spec)
        if len(inferred) >= 2:
            units = _deduplicate_repeat_units(inferred + units)
            spec["architecture_unit_inference_used"] = True

    if _architecture_is_homopolymer(spec) and len(units) > 1:
        # Homopolymers should not be expanded into multiple fragments.
        units = units[:1]
        spec["architecture_unit_trimmed"] = True

    # Keep common inorganic/non-polymer buffers out of repeat_units.
    non_polymer_markers = ["hydroxyapatite", "hap", "calcium phosphate", "silica", "graphene", "nanocellulose"]
    kept_units: list[dict[str, Any]] = []
    moved_nonpolymer: list[str] = []
    for unit in units:
        name = str(unit.get("extracted_name") or unit.get("label") or "").strip()
        low = name.lower()
        if any(m in low for m in non_polymer_markers):
            moved_nonpolymer.append(name)
        else:
            kept_units.append(unit)
    units = kept_units

    if moved_nonpolymer:
        existing = spec.get("non_polymer_components") or []
        if not isinstance(existing, list):
            existing = [str(existing)]
        spec["non_polymer_components"] = list(dict.fromkeys([str(x) for x in existing + moved_nonpolymer if str(x).strip()]))

    if requires_multi and len(units) < 2:
        spec["multi_unit_warning"] = (
            f"Architecture '{arch}' normally requires 2 or more polymeric units, "
            f"but only {len(units)} unit(s) were extracted/resolved."
        )

    spec["repeat_units"] = units
    return spec


def resolve_repeat_unit_smiles_from_dictionaries(
    spec: dict[str, Any],
    *,
    name_name_dict_path: str | os.PathLike[str] | None = None,
    name_smiles_csv_path: str | os.PathLike[str] | None = None,
    allow_fuzzy: bool = False,
    fuzzy_cutoff: float = 0.92,
    strict_smiles: bool = False,
    prefer_overall_polymer_if_available: bool = True,
    force_segment_level_schematic: bool = False,
) -> dict[str, Any]:

    spec = dict(spec or {})
    spec = _postprocess_repeat_units_by_architecture(spec)

    units = spec.get("repeat_units") or []
    if not isinstance(units, list):
        units = []

    if not units:
        fallback_name = spec.get("polymer_name") or spec.get("design_name") or "Proposed polymer design"
        if (not bool(spec.get("should_draw", False))) or _is_generic_polymer_name(fallback_name):
            spec["repeat_units"] = []
            spec["should_draw"] = False
            spec["confidence"] = "low"
            spec["mapping_warning"] = (
                "No specific polymer name was extracted; generic class terms "
                "such as polyester/polymer are not sent to the SMILES dictionary."
            )
            spec["name_mapping"] = {
                "name_name_dict_path": str(name_name_dict_path or ""),
                "name_smiles_csv_path": str(name_smiles_csv_path or ""),
                "rdkit_available": HAS_RDKIT,
                "valid_smiles_count": 0,
                "total_units": 0,
                "generic_name_guard": True,
            }
            return spec
        units = [{"extracted_name": fallback_name, "label": fallback_name, "role": "main proposed structure"}]

    filtered_units: list[Any] = []
    skipped_generic_names: list[str] = []
    for u in units:
        if isinstance(u, dict):
            rn = str(u.get("extracted_name") or u.get("structure_name") or u.get("name") or "").strip()
            lb = str(u.get("label") or "").strip()
        else:
            rn = str(u or "").strip()
            lb = rn

        if _is_generic_polymer_name(rn) and (not lb or _is_generic_polymer_name(lb)):
            skipped_generic_names.append(rn or lb)
            continue
        filtered_units.append(u)

    units = filtered_units
    if not units:
        spec["repeat_units"] = []
        spec["should_draw"] = False
        spec["confidence"] = "low"
        spec["mapping_warning"] = (
            "Only generic polymer-class names were extracted and were skipped: "
            + ", ".join([x for x in skipped_generic_names if x][:8])
        )
        spec["name_mapping"] = {
            "name_name_dict_path": str(name_name_dict_path or ""),
            "name_smiles_csv_path": str(name_smiles_csv_path or ""),
            "rdkit_available": HAS_RDKIT,
            "valid_smiles_count": 0,
            "total_units": 0,
            "generic_name_guard": True,
            "skipped_generic_names": skipped_generic_names,
        }
        return spec

    # ------------------------------------------------------------
    # Load dictionaries BEFORE any lookup.
    # ------------------------------------------------------------
    alias_map = load_name_name_dictionary(name_name_dict_path)
    smiles_map = load_name_smiles_dictionary(name_smiles_csv_path)

    alias_source_type = "none"
    if name_name_dict_path:
        try:
            alias_source_type = "keyword_txt_folder" if Path(name_name_dict_path).is_dir() else "mapping_file"
        except Exception:
            alias_source_type = "unknown"

    resolved_units: list[dict[str, Any]] = []
    valid_count = 0
    polymer_level_priority_used = False
    polymer_level_priority_skipped = False

    requires_multi = _architecture_requires_multiple_units(spec)
    independent_multi_component = _architecture_is_explicit_independent_multi_component(spec)

    # ------------------------------------------------------------
    # Ordered polymer-level priority.
    # Rule 1: try the overall polymer_name first.
    # Rule 2: if the overall name is found in the alias dictionary + SMILES CSV
    #         and is RDKit-valid, use it as the single drawable polymer.
    # Rule 3: only use multiple segments/components when the overall name fails,
    #         the user forces segment-level drawing, or the system is an explicit
    #         independent multi-polymer blend/composite.
    # ------------------------------------------------------------
    polymer_name = str(spec.get("polymer_name") or "").strip()
    overall_polymer_trials: list[dict[str, Any]] = []
    overall_polymer_priority_attempted = False
    overall_polymer_priority_blocked_reason = ""

    if (
        prefer_overall_polymer_if_available
        and not force_segment_level_schematic
        and not independent_multi_component
        and polymer_name
        and not _is_generic_polymer_name(polymer_name)
    ):
        overall_polymer_priority_attempted = True
        overall_unit, overall_polymer_trials = _try_resolve_overall_polymer_unit(
            polymer_name=polymer_name,
            alias_map=alias_map,
            smiles_map=smiles_map,
            allow_fuzzy=allow_fuzzy,
            fuzzy_cutoff=fuzzy_cutoff,
        )
        if overall_unit is not None:
            units = [overall_unit]
            polymer_level_priority_used = True
        else:
            polymer_level_priority_skipped = True
            overall_polymer_priority_blocked_reason = "overall_polymer_name_not_resolved_to_valid_dictionary_smiles"
        
            split_units: list[dict[str, Any]] = []
        
            # 1) 先拆 polymer_name，例如 PEG-PLA / PBS-b-PEG
            if _has_copolymer_or_component_separator(polymer_name):
                split_units = _split_full_polymer_name_after_overall_fail(
                    polymer_name,
                    spec.get("architecture", "unknown"),
                )
        
            # 2) 如果 polymer_name 拆不出两个，再拆 LLM 给的 repeat_units 里的 extracted_name
            #    例如 poly(ethylene glycol)-poly(lactic acid)
            if len(split_units) < 2:
                for u in units:
                    if not isinstance(u, dict):
                        continue
        
                    raw_u_name = str(
                        u.get("extracted_name")
                        or u.get("structure_name")
                        or u.get("name")
                        or u.get("label")
                        or ""
                    ).strip()
        
                    if raw_u_name and _has_copolymer_or_component_separator(raw_u_name):
                        tmp_units = _split_full_polymer_name_after_overall_fail(
                            raw_u_name,
                            spec.get("architecture", "unknown"),
                        )
                        if len(tmp_units) >= 2:
                            split_units = tmp_units
                            spec["split_after_failed_unit_lookup"] = True
                            break
        
            if len(split_units) >= 2:
                units = split_units
                spec["split_after_failed_overall_lookup"] = True
    else:
        polymer_level_priority_skipped = True
        if not prefer_overall_polymer_if_available:
            overall_polymer_priority_blocked_reason = "prefer_overall_polymer_if_available_false"
        elif force_segment_level_schematic:
            overall_polymer_priority_blocked_reason = "force_segment_level_schematic_true"
        elif independent_multi_component:
            overall_polymer_priority_blocked_reason = "explicit_independent_multi_component_architecture"
        elif not polymer_name:
            overall_polymer_priority_blocked_reason = "empty_polymer_name"
        elif _is_generic_polymer_name(polymer_name):
            overall_polymer_priority_blocked_reason = "generic_polymer_name"

    # ------------------------------------------------------------
    # Resolve final units.
    # If polymer-level priority succeeded, units contains only that
    # polymer-level structure. Otherwise, units are the architecture-aware
    # repeat_units / components.
    # ------------------------------------------------------------
    for unit in units:
        if not isinstance(unit, dict):
            unit = {"extracted_name": str(unit), "label": str(unit), "role": ""}
        else:
            unit = dict(unit)

        # If polymer-level priority already produced a validated unit,
        # preserve it and avoid re-resolving.
        if bool(unit.get("rdkit_valid")) and str(unit.get("smiles") or "").strip():
            resolved_units.append(unit)
            valid_count += 1
            continue

        raw_name = str(
            unit.get("extracted_name")
            or unit.get("structure_name")
            or unit.get("name")
            or unit.get("label")
            or spec.get("polymer_name")
            or ""
        ).strip()

        label = str(unit.get("label") or raw_name or "repeat unit").strip()
        role = str(unit.get("role") or "").strip()

        lookup_candidates = _collect_lookup_candidates_for_unit(unit, spec)
        if not lookup_candidates:
            lookup_candidates = [raw_name or label or str(spec.get("polymer_name") or "")]

        best_payload: dict[str, Any] | None = None
        candidate_trials: list[dict[str, Any]] = []

        for cand in lookup_candidates:
            canonical, smiles_record, alias_status, smiles_status = _resolve_candidate_to_record(
                cand,
                alias_map,
                smiles_map,
                allow_fuzzy=allow_fuzzy,
                fuzzy_cutoff=fuzzy_cutoff,
            )

            mapped_smiles = str((smiles_record or {}).get("smiles", "") or "").strip()
            db_name = str((smiles_record or {}).get("name", "") or "").strip()
            validation = validate_smiles_with_rdkit(mapped_smiles)
            rdkit_valid = bool(validation.get("valid"))

            trial = {
                "candidate": cand,
                "canonical_name": canonical,
                "database_name": db_name or canonical,
                "mapped_smiles": mapped_smiles,
                "rdkit_valid": rdkit_valid,
                "alias_status": alias_status,
                "smiles_status": smiles_status,
                "rdkit_validation": validation,
            }
            candidate_trials.append(trial)

            if rdkit_valid:
                best_payload = trial
                break

            if best_payload is None and mapped_smiles:
                best_payload = trial

        if best_payload is None:
            if candidate_trials:
                best_payload = candidate_trials[0]
            else:
                validation = validate_smiles_with_rdkit("")
                best_payload = {
                    "candidate": raw_name,
                    "canonical_name": raw_name,
                    "database_name": raw_name,
                    "mapped_smiles": "",
                    "rdkit_valid": False,
                    "alias_status": {"status": "empty_name", "match_method": "none"},
                    "smiles_status": {"status": "smiles_not_found", "match_method": "none", "score": 0.0},
                    "rdkit_validation": validation,
                }

        rdkit_valid = bool(best_payload.get("rdkit_valid"))
        if rdkit_valid:
            valid_count += 1

        mapped_smiles = str(best_payload.get("mapped_smiles", "") or "").strip()
        drawable_smiles = mapped_smiles if rdkit_valid else ""

        unit.update({
            "extracted_name": raw_name,
            "label": label,
            "role": role,
            "lookup_candidates": lookup_candidates,
            "selected_lookup_candidate": best_payload.get("candidate", ""),
            "canonical_name": best_payload.get("canonical_name", raw_name),
            "database_name": best_payload.get("database_name", best_payload.get("canonical_name", raw_name)),
            "mapped_smiles": mapped_smiles,
            "smiles": drawable_smiles,
            "rdkit_valid": rdkit_valid,
            "rdkit_validation": best_payload.get("rdkit_validation", validate_smiles_with_rdkit(mapped_smiles)),
            "name_resolution": {
                "alias_status": best_payload.get("alias_status", {}),
                "smiles_status": best_payload.get("smiles_status", {}),
                "candidate_trials": candidate_trials[:12],
            },
        })
        resolved_units.append(unit)

    spec["repeat_units"] = resolved_units
    spec["polymer_level_priority_used"] = polymer_level_priority_used
    spec["polymer_level_priority_skipped"] = polymer_level_priority_skipped
    spec["name_mapping"] = {
        "name_name_dict_path": str(name_name_dict_path or ""),
        "alias_source_type": alias_source_type,
        "name_smiles_csv_path": str(name_smiles_csv_path or ""),
        "alias_entries": len(alias_map),
        "smiles_entries": len(smiles_map),
        "allow_fuzzy": allow_fuzzy,
        "fuzzy_cutoff": fuzzy_cutoff,
        "rdkit_available": HAS_RDKIT,
        "valid_smiles_count": valid_count,
        "total_units": len(resolved_units),
        "generic_name_guard": True,
        "keyword_folder_match_required": bool(alias_map),
        "polymer_level_priority_used": polymer_level_priority_used,
        "polymer_level_priority_skipped": polymer_level_priority_skipped,
        "prefer_overall_polymer_if_available": prefer_overall_polymer_if_available,
        "force_segment_level_schematic": force_segment_level_schematic,
        "overall_polymer_priority_attempted": overall_polymer_priority_attempted,
        "overall_polymer_priority_blocked_reason": overall_polymer_priority_blocked_reason,
        "overall_polymer_trials": overall_polymer_trials[:8],
        "architecture_requires_multiple_units": requires_multi,
        "independent_multi_component": independent_multi_component,
        "split_after_failed_overall_lookup": bool(spec.get("split_after_failed_overall_lookup", False)),
    }

    if valid_count > 0:
        spec["should_draw"] = True
        spec["confidence"] = spec.get("confidence") or "medium"
    elif strict_smiles:
        spec["should_draw"] = False
        spec["confidence"] = "low"
        spec["mapping_error"] = "No extracted polymer name could be resolved to an RDKit-valid dictionary SMILES."
    else:
        spec["should_draw"] = bool(spec.get("should_draw", True))
        spec["confidence"] = "low"
        spec["mapping_warning"] = "No RDKit-valid dictionary SMILES found; schematic fallback will be used."

    if requires_multi and not polymer_level_priority_used and len(resolved_units) < 2:
        spec["multi_unit_warning"] = (
            f"Architecture '{spec.get('architecture', 'unknown')}' normally needs 2 or more polymeric units when segment-level fallback is used; "
            f"only {len(resolved_units)} unit(s) were resolved. Check alias dictionary and name-SMILES CSV."
        )

    return spec
    

# -----------------------------------------------------------------------------
# Extraction prompt: LLM extracts names only, not SMILES
# -----------------------------------------------------------------------------
EXTRACT_SYSTEM_PROMPT = """
You are a senior polymer chemist. Your task is to extract conservative, database-matchable POLYMER NAMES from a Design-mode biodegradable polymer report.

Return JSON only. Do not include prose, Markdown, citations, code fences, SMILES, SMARTS, InChI, or molecular formulas.

Important architecture policy:
- First classify the material architecture as one of: homopolymer, random copolymer, block copolymer, graft copolymer, segmented copolymer, blend/composite, network, modified polymer, or unknown.
- Priority rule: extract the explicit overall polymer-level name as polymer_name. The resolver will first perform strict full-name matching against the dictionary/SMILES database. Examples: PBAT / poly(butylene adipate-co-terephthalate), PBS-b-PEG, and poly(butylene succinate-co-isosorbide succinate).
- Do NOT decompose an explicit overall copolymer name in the extraction step. If the complete name is not found later by strict full-name lookup, the resolver may then split names containing -co-, -b-, -g-, /, +, or : into polymer-level components.
- Extract multiple polymer-level entries directly only when the report explicitly describes independent polymer components such as blends/composites, or when the report itself explicitly lists component polymers as separate drawable structures.
- For polymer + inorganic/small-molecule filler systems, extract the polymeric structure(s) into repeat_units and put fillers/additives such as hydroxyapatite, calcium phosphate, silica, catalysts, solvents, and buffer particles into non_polymer_components.
- Do NOT extract generic classes such as polymer, polyester, copolyester, biodegradable polymer, matrix, composite, segment, unit, linkage, hard segment, or soft segment as drawable polymers. Do NOT extract monomer/repeat-fragment names such as butylene adipate, butylene terephthalate, succinate, adipate, or terephthalate as repeat_units.
- Put non-polymer functional motifs, linkages, unspecified comonomers, catalysts, solvents, and exact-structure-unspecified modifiers into functional_groups or non_polymer_components, not into repeat_units.
- If no concrete polymer name is present, set should_draw=false.
""".strip()


def build_repeat_unit_extraction_prompt(
    *,
    report_text: str,
    original_query: str = "",
    idea_title: str = "",
    idea_mechanism: str = "",
) -> str:
    trimmed_report = str(report_text or "")[:16000]
    return f"""
Read the Design-mode report and extract the most important drawable, database-matchable polymer names.

Original user query:
{original_query}

Idea title:
{idea_title}

Idea mechanism:
{idea_mechanism}

Report text:
{trimmed_report}

STRICT RULES:
1. Extract only specific, database-matchable polymer names explicitly supported by the report.
2. The repeat_units field is for drawable polymer structures only.
3. First identify the architecture: homopolymer, random copolymer, block copolymer, graft copolymer, segmented copolymer, blend/composite, network, modified polymer, or unknown.
4. For homopolymers, use the highest polymer-level name available in the report and usually return one repeat_units entry.
5. For copolymers, block copolymers, graft copolymers, and segmented polymers with an explicit overall name, keep the overall design name as polymer_name and usually put that complete polymer name into repeat_units. Do not split it during extraction.
6. If the complete overall name is not found later by strict dictionary/SMILES lookup, the resolver will split connector-containing names such as PBS-b-PEG, PBS/PEG, PBS-co-PEG, PEG-modified PBS, or poly(butylene succinate)-b-poly(ethylene glycol) into polymer-level components as fallback.
7. If splitting is needed for poly(A-co-B)-type names, the fallback components must still be polymer-level names such as poly(butylene succinate), PBS, PEG, or poly(isosorbide succinate), not bare monomer/fragment names.
8. Do NOT put monomers, small molecules, solvents, catalysts, inorganic fillers, functional groups, linkers, generic material classes, or bare repeat-fragment names into repeat_units. Use polymer names only, e.g., poly(butylene adipate), poly(butylene terephthalate), poly(ethylene glycol).
9. Put non-polymer functional groups, linkages, unspecified comonomers, inorganic fillers, pH buffers, and exact-structure-unspecified modifiers into functional_groups or non_polymer_components.
10. Do not invent exact molar ratios, stereochemistry, molecular weight, or end groups if the report does not state them.
11. If no concrete polymer name is present, set should_draw=false and return an empty repeat_units list.
12. Use English labels. Keep labels short enough for a figure.

Return exactly this JSON schema:
{{
  "should_draw": true,
  "confidence": "high|medium|low",
  "polymer_name": "short overall polymer/design name",
  "architecture": "homopolymer|random copolymer|block copolymer|graft copolymer|segmented copolymer|blend/composite|network|modified polymer|unknown",
  "repeat_units": [
    {{
      "extracted_name": "database-matchable polymer-level name only, e.g., poly(butylene adipate), PBA, PBS, PEG; no SMILES and no bare monomer/repeat-fragment names",
      "label": "short figure label, e.g., PBS segment / PEG segment",
      "role": "one short role, e.g., base polyester block / hydrophilic soft block / grafted polymer segment"
    }}
  ],
  "functional_groups": [
    {{
      "name": "functional group, linkage, or unspecified comonomer not used for polymer-SMILES lookup",
      "role": "short role, e.g., hydrogen-bonding linkage / exact structure unspecified"
    }}
  ],
  "composition_note": "n:m, x:y, PBS:PEG ratio, TBD, or architecture note; do not invent numbers",
  "non_polymer_components": ["short additive/filler labels if relevant, e.g., HAp nanoparticles"],
  "caption": "one concise figure caption beginning with: Representative polymer-structure schematic of ..."
}}
""".strip()


def extract_repeat_unit_spec(
    *,
    report_text: str,
    original_query: str = "",
    idea_title: str = "",
    idea_mechanism: str = "",
    llm_callable: Callable[..., str] | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Use DeepSeek-R1 to extract conservative structure names only."""
    llm = llm_callable or _default_deepseek_callable()
    if llm is None:
        return {
            "should_draw": False,
            "confidence": "low",
            "polymer_name": idea_title or "Unknown design",
            "architecture": "unknown",
            "repeat_units": [],
            "composition_note": "LLM callable unavailable.",
            "non_polymer_components": [],
            "caption": "No repeat-unit figure generated because the LLM callable is unavailable.",
            "error": "missing_llm_callable",
        }

    prompt = build_repeat_unit_extraction_prompt(
        report_text=report_text,
        original_query=original_query,
        idea_title=idea_title,
        idea_mechanism=idea_mechanism,
    )
    raw = _call_llm(llm, prompt, EXTRACT_SYSTEM_PROMPT, temperature=temperature)
    obj = _extract_json_object(raw)
    if not obj:
        return {
            "should_draw": False,
            "confidence": "low",
            "polymer_name": idea_title or "Unknown design",
            "architecture": "unknown",
            "repeat_units": [],
            "composition_note": "Failed to parse DeepSeek JSON.",
            "non_polymer_components": [],
            "caption": "No repeat-unit figure generated because structure-name extraction failed.",
            "raw_response": _strip_think_tags(raw)[:2000],
            "error": "json_parse_failed",
        }

    obj.setdefault("should_draw", False)
    obj.setdefault("confidence", "low")
    obj.setdefault("polymer_name", idea_title or "Proposed polymer design")
    obj.setdefault("architecture", "unknown")
    obj.setdefault("repeat_units", [])
    obj.setdefault("composition_note", "TBD")
    obj.setdefault("non_polymer_components", [])
    obj.setdefault("caption", f"Representative repeat-unit schematic of {obj.get('polymer_name') or idea_title}.")
    obj.setdefault("functional_groups", [])
    
    if not isinstance(obj.get("functional_groups"), list):
        obj["functional_groups"] = []
    if not isinstance(obj.get("repeat_units"), list):
        obj["repeat_units"] = []
    if not isinstance(obj.get("non_polymer_components"), list):
        obj["non_polymer_components"] = []

    # Architecture-aware correction before generic filtering. This lets systems like
    # PBS-b-PEG or poly(butylene succinate-co-isosorbide succinate) become multi-unit
    # schematics instead of collapsing to a single overall copolymer name.
    obj = _postprocess_repeat_units_by_architecture(obj)

    # Hard safety guard: ignore any LLM-provided SMILES-like fields and discard
    # generic class-level names before they can reach dictionary lookup.
    cleaned_units = []
    skipped_generic_names: list[str] = []
    for unit in obj.get("repeat_units") or []:
        if not isinstance(unit, dict):
            unit = {"extracted_name": str(unit), "label": str(unit), "role": ""}
        unit = dict(unit)
        for forbidden in ["smiles", "SMILES", "smile", "SMARTS", "inchi", "InChI"]:
            unit.pop(forbidden, None)
        unit.setdefault("extracted_name", unit.get("label") or obj.get("polymer_name") or "")
        unit.setdefault("label", unit.get("extracted_name") or "repeat unit")
        unit.setdefault("role", "")

        extracted = str(unit.get("extracted_name") or "").strip()
        label = str(unit.get("label") or "").strip()
        if _is_generic_polymer_name(extracted) and _is_generic_polymer_name(label):
            skipped_generic_names.append(extracted or label)
            continue
        cleaned_units.append(unit)

    obj["repeat_units"] = cleaned_units
    if skipped_generic_names:
        obj["generic_skipped_names"] = skipped_generic_names
    if not cleaned_units:
        obj["should_draw"] = False
        obj["confidence"] = "low"
        obj["composition_note"] = (
            "No specific polymer/repeat-unit name was extracted; generic polymer-class terms were skipped."
        )

    return obj


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def _prepare_mol_from_smiles(smiles: str):
    if not HAS_RDKIT or not smiles:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)  # type: ignore[union-attr]
        if mol is None:
            return None
        try:
            Chem.rdDepictor.Compute2DCoords(mol)  # type: ignore[union-attr]
        except Exception:
            pass
        return mol
    except Exception:
        return None


def _render_rdkit_grid(spec: dict[str, Any], out_png: str) -> bool:
    repeat_units = spec.get("repeat_units") or []
    mols = []
    legends = []
    for unit in repeat_units:
        if not isinstance(unit, dict):
            continue
        # Use only dictionary-derived, RDKit-validated `smiles`.
        mol = _prepare_mol_from_smiles(unit.get("smiles", ""))
        if mol is None:
            continue

        label = str(
            unit.get("label")
            or unit.get("database_name")
            or unit.get("canonical_name")
            or "repeat unit"
        ).strip()
        
        role = str(unit.get("role") or "").strip()
        
        # 长名称自动换行，而不是直接截断
        label = textwrap.fill(label, width=32)
        role = textwrap.fill(role, width=36)
        
        legend = label if not role else f"{label}\n{role}"

        
        mols.append(mol)
        legends.append(legend)

    if not mols:
        return False

    try:
        img = Draw.MolsToGridImage(  # type: ignore[union-attr]
            mols,
            molsPerRow=min(2, max(1, len(mols))),
            subImgSize=(700, 420),
            legends=legends,
            useSVG=False,
            returnPNG=False,
        )
        if hasattr(img, "save"):
            img.save(out_png)
            return os.path.exists(out_png)
        data = getattr(img, "data", None)
        if data:
            with open(out_png, "wb") as f:
                f.write(data)
            return os.path.exists(out_png)
    except Exception:
        return False
    return False


def _render_schematic_fallback(spec: dict[str, Any], out_png: str) -> bool:
    """Render an architecture-level fallback schematic when exact SMILES are absent."""
    units = spec.get("repeat_units") or []
    if not units:
        units = [{"label": spec.get("polymer_name", "Proposed repeat unit"), "role": spec.get("architecture", "architecture-level schematic")}]

    labels = []
    roles = []
    for unit in units[:4]:
        if isinstance(unit, dict):
            labels.append(str(unit.get("label") or unit.get("canonical_name") or unit.get("extracted_name") or "Unit"))
            role = str(unit.get("role") or "")
            if unit.get("canonical_name") and unit.get("canonical_name") != unit.get("extracted_name"):
                role = (role + f"; mapped: {unit.get('canonical_name')}").strip("; ")
            if unit.get("rdkit_valid") is False and unit.get("mapped_smiles"):
                role = (role + "; invalid SMILES").strip("; ")
            roles.append(role)
        else:
            labels.append(str(unit))
            roles.append("")

    title = f"Representative repeat-unit schematic: {spec.get('polymer_name', 'proposed polymer')}"
    arch = f"Architecture: {spec.get('architecture', 'unknown')} | Composition: {spec.get('composition_note', 'TBD')}"

    if HAS_MATPLOTLIB:
        try:
            fig_w = max(7.0, 2.2 * len(labels))
            fig, ax = plt.subplots(figsize=(fig_w, 3.2), dpi=300)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 4)
            ax.axis("off")
            ax.text(5, 3.65, title, ha="center", va="center", fontsize=11, fontweight="bold")
            ax.text(5, 3.25, arch, ha="center", va="center", fontsize=8)

            n = len(labels)
            start_x = 1.0
            end_x = 9.0
            gap = (end_x - start_x) / max(n, 1)
            box_w = min(1.65, gap * 0.75)
            y = 1.75

            for i, label in enumerate(labels):
                cx = start_x + gap * (i + 0.5)
                rect = plt.Rectangle((cx - box_w / 2, y - 0.45), box_w, 0.9, fill=False, linewidth=1.6)
                ax.add_patch(rect)
                ax.text(cx, y + 0.12, textwrap.fill(label, 16), ha="center", va="center", fontsize=8)
                if roles[i]:
                    ax.text(cx, y - 0.78, textwrap.fill(roles[i], 20), ha="center", va="top", fontsize=7)
                if i < n - 1:
                    ax.annotate("", xy=(cx + gap * 0.43, y), xytext=(cx + box_w / 2 + 0.05, y), arrowprops=dict(arrowstyle="-", lw=1.2))

            ax.text(0.6, y, "[*]", ha="center", va="center", fontsize=10)
            ax.text(9.4, y, "[*]", ha="center", va="center", fontsize=10)
            fig.savefig(out_png, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            return os.path.exists(out_png)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass

    if HAS_PIL:
        try:
            img = Image.new("RGB", (1600, 520), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((40, 35), title, fill="black", font=font)
            draw.text((40, 70), arch, fill="black", font=font)
            x = 90
            y = 210
            for i, label in enumerate(labels):
                draw.rectangle((x, y, x + 280, y + 90), outline="black", width=3)
                draw.text((x + 18, y + 20), label[:45], fill="black", font=font)
                if roles[i]:
                    draw.text((x + 18, y + 112), roles[i][:60], fill="black", font=font)
                if i < len(labels) - 1:
                    draw.line((x + 280, y + 45, x + 360, y + 45), fill="black", width=3)
                x += 360
            img.save(out_png)
            return os.path.exists(out_png)
        except Exception:
            return False

    return False


def render_repeat_unit_figure(
    *,
    spec: dict[str, Any],
    output_dir: str | os.PathLike[str],
    base_name: str = "repeat_unit",
) -> dict[str, Any]:
    """Render repeat-unit figure and return metadata."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_base = _sanitize_filename(base_name)
    png_path = out_dir / f"{safe_base}_repeat_unit.png"
    spec_path = out_dir / f"{safe_base}_repeat_unit_spec.json"

    rendered = False
    renderer = "none"

    if bool(spec.get("should_draw", False)):
        if HAS_RDKIT:
            rendered = _render_rdkit_grid(spec, str(png_path))
            if rendered:
                renderer = "rdkit_dictionary_smiles"
        if not rendered:
            rendered = _render_schematic_fallback(spec, str(png_path))
            if rendered:
                renderer = "schematic_fallback"

    meta = {
        "rendered": bool(rendered),
        "renderer": renderer,
        "image_path": str(png_path) if rendered else "",
        "spec_path": str(spec_path),
        "spec": spec,
    }

    try:
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return meta


# -----------------------------------------------------------------------------
# Markdown insertion
# -----------------------------------------------------------------------------
def _make_markdown_image_block(image_path: str, caption: str, spec: dict[str, Any]) -> str:
    caption = str(caption or spec.get("caption") or "Representative repeat-unit schematic.").strip()
    path_for_md = str(image_path).replace("\\", "/")
    confidence = str(spec.get("confidence", "low"))
    composition_note = str(spec.get("composition_note", "TBD"))
    architecture = str(spec.get("architecture", "unknown"))
    mapping = spec.get("name_mapping") or {}
    valid_count = mapping.get("valid_smiles_count", 0)
    total_units = mapping.get("total_units", len(spec.get("repeat_units") or []))

    return (
        "\n\n## Graphical Repeat-Unit Representation\n\n"
        f"![Representative repeat-unit schematic]({path_for_md})\n\n"
        f"**Figure.** {caption}\n\n"
        f"Structure confidence: {confidence}. Architecture: {architecture}. Composition note: {composition_note}. "
        f"Dictionary/RDKit validation: {valid_count}/{total_units} mapped structure(s) validated.\n"
    )

def insert_repeat_unit_figure_into_report(
    report_text: str,
    *,
    image_path: str,
    caption: str,
    spec: dict[str, Any],
    placement: str = "after_abstract",
) -> str:
    """Insert or replace the generated repeat-unit figure block in Markdown report text."""
    if not image_path or not os.path.exists(image_path):
        return report_text

    # 始终生成完整小节。若旧小节存在，则整段替换，避免重复 caption / validation。
    block = _make_markdown_image_block(image_path, caption, spec).strip()

    # 1. 如果已经存在 Graphical 小节：删除旧小节，替换为新的完整 block
    # 小节范围：从 Graphical 标题开始，到下一个 Markdown heading 或数字编号章节前结束。
    existing_section_pat = re.compile(
        r"(?ims)"
        r"^##\s*Graphical Repeat-Unit Representation\s*"
        r".*?"
        r"(?=^##\s+|\n\d+\.\s+[A-Z]|\Z)"
    )

    if existing_section_pat.search(report_text):
        return existing_section_pat.sub(block + "\n\n", report_text, count=1).strip() + "\n"

    # 兼容旧 DOCX/Markdown 转换后标题 # 可能丢失的情况：
    # Graphical Repeat-Unit Representation 不是 ## heading，而是普通行。
    existing_plain_pat = re.compile(
        r"(?ims)"
        r"^Graphical Repeat-Unit Representation\s*"
        r".*?"
        r"(?=^##\s+|\n\d+\.\s+[A-Z]|\Z)"
    )

    if existing_plain_pat.search(report_text):
        return existing_plain_pat.sub(block + "\n\n", report_text, count=1).strip() + "\n"

    # 2. 没有旧小节时：按 placement 插入
    full_block = "\n\n" + block + "\n\n"

    if placement == "after_title":
        m = re.search(r"(?m)^#\s+.+$", report_text)
        if m:
            return report_text[:m.end()] + full_block + report_text[m.end():].lstrip()

    m_abs = re.search(
        r"(?is)(##\s*Abstract\s*.*?)(?=\n##\s+\d+\.|\n##\s+[A-Z]|\n\d+\.\s+[A-Z]|\Z)",
        report_text,
    )
    if m_abs:
        return report_text[:m_abs.end()].rstrip() + full_block + report_text[m_abs.end():].lstrip()

    m_title = re.search(r"(?m)^#\s+.+$", report_text)
    if m_title:
        return report_text[:m_title.end()] + full_block + report_text[m_title.end():].lstrip()

    return block + "\n\n" + report_text
    
def enrich_design_report_with_repeat_unit_figure(
    *,
    report_text: str,
    output_dir: str | os.PathLike[str],
    original_query: str = "",
    idea_title: str = "",
    idea_mechanism: str = "",
    idea_tag: str = "idea",
    llm_callable: Callable[..., str] | None = None,
    name_name_dict_path: str | os.PathLike[str] | None = None,
    name_smiles_csv_path: str | os.PathLike[str] | None = None,
    placement: str = "after_abstract",
    allow_fuzzy_name_match: bool = False,
    fuzzy_cutoff: float = 0.92,
    strict_smiles: bool = False,
    prefer_overall_polymer_if_available: bool = True,
    force_segment_level_schematic: bool = False,
    verbose: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Main callable used by other scripts.

    Key change from the original version:
    - LLM extracts names only.
    - SMILES are loaded exclusively from user dictionaries/CSV.
    - RDKit validates mapped SMILES before chemical rendering.

    Returns:
        (updated_markdown_report, metadata)
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    spec = extract_repeat_unit_spec(
        report_text=report_text,
        original_query=original_query,
        idea_title=idea_title,
        idea_mechanism=idea_mechanism,
        llm_callable=llm_callable,
        temperature=0.0,
    )

    try:
        spec = resolve_repeat_unit_smiles_from_dictionaries(
            spec,
            name_name_dict_path=name_name_dict_path,
            name_smiles_csv_path=name_smiles_csv_path,
            allow_fuzzy=allow_fuzzy_name_match,
            fuzzy_cutoff=fuzzy_cutoff,
            strict_smiles=strict_smiles,
            prefer_overall_polymer_if_available=prefer_overall_polymer_if_available,
            force_segment_level_schematic=force_segment_level_schematic,
        )
    except Exception as e:
        spec["should_draw"] = bool(spec.get("should_draw", False)) and not strict_smiles
        spec["confidence"] = "low"
        spec["mapping_error"] = str(e)
        spec["name_mapping"] = {
            "name_name_dict_path": str(name_name_dict_path or ""),
            "name_smiles_csv_path": str(name_smiles_csv_path or ""),
            "error": str(e),
            "rdkit_available": HAS_RDKIT,
        }

    meta = render_repeat_unit_figure(
        spec=spec,
        output_dir=fig_dir,
        base_name=f"{idea_tag}_{idea_title or 'repeat_unit'}",
    )

    updated = report_text
    if meta.get("rendered"):
        updated = insert_repeat_unit_figure_into_report(
            report_text,
            image_path=meta["image_path"],
            caption=spec.get("caption", "Representative repeat-unit schematic."),
            spec=spec,
            placement=placement,
        )

    if verbose:
        status = "generated" if meta.get("rendered") else "skipped"
        mapping = spec.get("name_mapping") or {}
        print(
            f"[repeat_unit_figure] {status}: {meta.get('image_path', '')} | "
            f"renderer={meta.get('renderer')} | "
            f"valid_smiles={mapping.get('valid_smiles_count', 0)}/{mapping.get('total_units', 0)}",
            flush=True,
        )
        if spec.get("mapping_error"):
            print(f"[repeat_unit_figure] mapping_error: {spec.get('mapping_error')}", flush=True)
        if spec.get("mapping_warning"):
            print(f"[repeat_unit_figure] mapping_warning: {spec.get('mapping_warning')}", flush=True)

    return updated, meta


__all__ = [
    "extract_repeat_unit_spec",
    "load_name_name_dictionary",
    "load_keyword_folder_dictionary",
    "load_name_smiles_dictionary",
    "resolve_repeat_unit_smiles_from_dictionaries",
    "_architecture_requires_multiple_units",
    "_postprocess_repeat_units_by_architecture",
    "_has_copolymer_or_component_separator",
    "_split_full_polymer_name_after_overall_fail",
    "validate_smiles_with_rdkit",
    "render_repeat_unit_figure",
    "insert_repeat_unit_figure_into_report",
    "enrich_design_report_with_repeat_unit_figure",
]
