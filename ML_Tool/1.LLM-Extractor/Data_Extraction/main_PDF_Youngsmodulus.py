#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python main-PDF-Youngsmodulus.py

import os
import shutil
import pdfplumber

from API_YoungsModulus import call_deepseek_llm
from contains_keywords_youngs import contains_keywords
from Clean import ParagraphParser  # Used for filtering

# ========== Path Configuration ==========
# Fill in the root directory for "split 5-page" PDFs here
INPUT_FOLDER = r"Data/Data_split"

PROCESSED_FOLDER = os.path.join(INPUT_FOLDER, "processed_pdfs")   # Successfully processed
WASTED_PDFS = os.path.join(INPUT_FOLDER, "wasted_pdfs")           # Unusable documents
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(WASTED_PDFS, exist_ok=True)

# ========== Keyword Dictionary (Polymers + Mechanical Properties) ==========
KEYWORD_DICT = {
    "Polymers": [
        # General polymers and copolymers
        "polymer", "copolymer", "blend", "biopolymer",
        # Common commercial or synthetic polymers
        "PLA", "polylactic acid", "PCL", "polycaprolactone",
        "PET", "polyethylene terephthalate", "PMMA", "polymethyl methacrylate",
        "PU", "polyurethane", "PA", "polyamide", "nylon",
        "PVC", "polyvinyl chloride", "PVA", "polyvinyl alcohol",
        "PAN", "polyacrylonitrile", "PVP", "polyvinylpyrrolidone",
        "PDMS", "polydimethylsiloxane", "PC", "polycarbonate",
        "PBS", "polybutylene succinate", "PGA", "poly(γ-glutamic acid)",
        "PE", "polyethylene", "PP", "polypropylene", "polyester",
        # Bio-based/Natural polymers
        "GelMA", "gelatin methacrylate", "gelatin", "collagen",
        "chitosan", "alginate", "sodium alginate", "cellulose",
        "nanocellulose", "pectin", "lignin", "starch", "hyaluronic acid",
        "silk fibroin", "polyethylene glycol", "polydopamine",
        "polyacrylamide"
    ],
    "Additives or Modifiers": [
        # Basic functional additives
        "additive", "modifier", "plasticizer", "compatibilizer", "filler", "blend",
        "hybrid", "nanocomposite", "composite",
        # Inorganic nanomaterials
        "nanoparticle", "nanofiller", "nanoclay", "TiO2", "SiO2", "ZnO", "CaCO3",
        "clay", "montmorillonite", "halloysite", "bentonite",
        # Organic/Polymer blends
        "PBAT", "PEG", "PHA", "PBSA", "PPC", "EVA", "PLA-g-MA",
        # Bio-based/Natural materials
        "cellulose nanocrystal", "microcrystalline cellulose", "hemicellulose",
        "soy protein", "wheat bran", "rice husk",
        # Carbon-based materials
        "CNT", "carbon nanotube", "carbon black", "graphene", "graphene oxide",
        "reduced graphene oxide",
        # Fibers and reinforcements
        "fiber", "natural fiber", "glass fiber", "bamboo fiber", "hemp fiber",
        "basalt fiber", "jute fiber", "kenaf fiber",
        # Plasticizers
        "glycerol", "triacetin", "citrate", "ATBC", "TEC", "tributyl citrate",
        "polyethylene glycol",
        # Blending and compatibilization
        "blending", "blended", "copolymerized", "reactive compatibilization",
        "immiscible", "miscible",
        # Other additives
        "antioxidant", "nucleating agent", "chain extender", "crosslinker",
        "UV stabilizer", "thermal stabilizer", "fire retardant", "flame retardant"
    ],
    "mechanical properties": [
        "tensile strength", "breaking strength", "tensile properties",
        "elongation at break", "breaking elongation",
        "young's modulus", "tensile modulus",
        "flexural modulus", "bending modulus", "flexural stiffness",
        "impact strength", "impact toughness",
        "stress-strain", "mechanical behavior",
        "hardness", "shore hardness", "rockwell", "durometer",
    ],
    "Glass Transition": ["glass transition", "Tg"],
    "Melting Point": ["melting point", "melting temperature", "Tm"]
}


def extract_pdf_with_layout(pdf_path: str) -> str:
    """
    Extracts PDF text (excluding tables to avoid Ascii85 issues).
    Adds "=== Page n ===" before each page to facilitate debugging & few-shot learning.
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
                full_text.append(f"\n=== Page {page.page_number} ===\n{text}")
            except Exception as e:
                full_text.append(f"\n⚠️ Extraction failed for page {page.page_number}: {e}")
    return "\n".join(full_text)


def process_pdf(pdf_path: str):
    """
    Process a single PDF:
    - Extract text
    - Clean paragraphs
    - Filter keywords
    - Call LLM, write to extracted_output
    - Move original PDF to processed_pdfs / wasted_pdfs
    """
    filename = os.path.basename(pdf_path)
    # Create extracted_output in the directory where the current PDF is located
    out_dir = os.path.join(os.path.dirname(pdf_path), "extracted_output")
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, filename.replace(".pdf", ".txt"))

    # Skip if already processed
    if os.path.exists(txt_path):
        print(f"⏩ Already processed, skipping: {filename}")
        shutil.move(pdf_path, os.path.join(PROCESSED_FOLDER, filename))
        return

    # Step 0: PDF Text Extraction
    try:
        raw = extract_pdf_with_layout(pdf_path)
    except Exception as e:
        print(f"❌ Extraction failed for {filename}: {e}")
        return

    # Step 0.5: Paragraph Parsing & Cleaning
    parser = ParagraphParser(name=filename, debug=False)
    parser.parse(raw)
    if not parser.is_valid():
        # If you want to discard these directly, they are moved to wasted_pdfs
        print(f"⚠️ {filename} determined as invalid text after parsing, skipping")
        shutil.move(pdf_path, os.path.join(WASTED_PDFS, filename))
        return
    text = parser.text.strip()

    # Step 1: Check if it contains "Polymers"
    found_polymers, _, matched_polymer_words = contains_keywords(
        text,
        {"Polymers": KEYWORD_DICT["Polymers"]}
    )
    if not found_polymers:
        print(f"⏩ {filename} at {pdf_path} contains no Polymer keywords, moving to wasted_pdfs")
        shutil.move(pdf_path, os.path.join(WASTED_PDFS, filename))
        return
    else:
        matched = matched_polymer_words.get("Polymers", [])
        if matched:
            print(f"📌 {filename} at {pdf_path} hit Polymer keywords: {', '.join(matched)}")

    # Step 2: Check if it contains mechanical properties keywords
    mech_dict = {"mechanical properties": KEYWORD_DICT["mechanical properties"]}
    found_mech, matched_categories, matched_words = contains_keywords(text, mech_dict)

    if not found_mech:
        print(f"⏩ {filename} at {pdf_path} contains Polymer keywords but no Mechanical Properties keywords, moving to wasted_pdfs")
        shutil.move(pdf_path, os.path.join(WASTED_PDFS, filename))
        return
    else:
        print(f"📈 {filename} at {pdf_path} hit Mechanical Properties keywords:"
              f"{', '.join(matched_words.get('mechanical properties', []))}")
        for cat, words in matched_words.items():
            print(f"    - {cat}: {', '.join(words)}")

    # Step 3: Call LLM to extract parameters
    prompt = f"📃 Text containing relevant parameters:\n{text}"
    try:
        ans = call_deepseek_llm(prompt)
        if not ans or not ans.strip():
            print(f"⚠️ No API response for {filename} at {pdf_path}, skipping")
            return
        content = f"🧾 Extraction Result:\n{ans.strip()}\n\n"
    except Exception as e:
        print(f"❌ Error processing {filename} at {pdf_path}: {e}")
        return

    # Step 4: Write txt + Move pdf
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    shutil.move(pdf_path, os.path.join(PROCESSED_FOLDER, filename))
    print(f"✅ Extraction complete: {filename}")
    print(f"📂 Original path: {pdf_path}")
    print(f"📄 Output path: {txt_path}")


def main():
    """
    Main entry point: Traverses all PDFs in INPUT_FOLDER and calls process_pdf
    """
    print(f"🔍 Starting PDF traversal and processing. Root directory: {INPUT_FOLDER}\n")

    for root, dirs, files in os.walk(INPUT_FOLDER):
        # Filter out subdirectories that do not need recursion
        dirs[:] = [
            d for d in dirs
            if d not in (
                os.path.basename(PROCESSED_FOLDER),
                os.path.basename(WASTED_PDFS),
                "extracted_output",
            )
        ]

        for fn in files:
            if fn.lower().endswith('.pdf') and not fn.startswith('._'):
                pdf_path = os.path.join(root, fn)
                print(f"\n==============================")
                print(f"📄 Processing: {pdf_path}")
                print(f"==============================")
                process_pdf(pdf_path)

    print("\n🎉 All traversals completed.")


if __name__ == "__main__":
    main()