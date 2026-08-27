#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import time
import re
import requests
import pandas as pd
from urllib.parse import quote


# ==========================
# User Configuration Area
# ==========================

# 🔑 Your Wiley TDM Token (Must be filled in)
WILEY_TDM_TOKEN = "your_token"

# 📄 Input CSV file containing DOIs (Must include a DOI column)
DOI_CSV = "your_doi.csv"

# 📁 Output PDF directory
OUTPUT_DIR = "./wiley-pdfs"

# Column name for DOI
DOI_COLNAME = "DOI"

# API Base URL
BASE_URL = "https://api.wiley.com/onlinelibrary/tdm/v1/articles"


# ==========================
# Helper Functions
# ==========================

def read_dois_from_csv(path, colname="DOI"):
    """
    Reads the DOI column from CSV, removes duplicates, and returns a list.
    """
    df = pd.read_csv(path)

    if colname not in df.columns:
        raise ValueError(f"❌ Column '{colname}' not found in CSV file.")

    dois = df[colname].dropna().unique()
    return [doi.strip() for doi in dois]


def sanitize_filename(name: str) -> str:
    """
    Sanitizes filenames by replacing illegal characters with underscores.
    """
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", name.strip())


def download_one(doi: str, out_dir: str):
    """
    Downloads a PDF for a single DOI.
    """
    url = f"{BASE_URL}/{quote(doi, safe='')}"
    headers = {
        "Wiley-TDM-Client-Token": WILEY_TDM_TOKEN,
        "User-Agent": "Wiley-TDM-Downloader/1.0",
        "Accept": "application/pdf"
    }

    try:
        r = requests.get(url, headers=headers, timeout=90)
    except Exception as e:
        print(f"⚠️ Request failed: {doi} | Error: {e}")
        return

    content_type = r.headers.get("Content-Type", "").lower()

    if r.status_code == 200 and "pdf" in content_type:
        fname = sanitize_filename(doi) + ".pdf"
        fpath = os.path.join(out_dir, fname)

        with open(fpath, "wb") as f:
            f.write(r.content)

        print(f"✅ Download successful: {doi}")
    else:
        print(f"❌ Download failed: {doi} | Status Code: {r.status_code}")
        print("Response content summary:", r.text[:200])


def batch_download(dois, out_dir=OUTPUT_DIR):
    """
    Batch downloads a list of DOIs.
    """
    os.makedirs(out_dir, exist_ok=True)

    for i, doi in enumerate(dois, 1):
        print(f"\n[{i}/{len(dois)}] Downloading: {doi}")
        download_one(doi, out_dir)
        time.sleep(6)  # Avoid hitting rate limits


# ==========================
# Main Entry Point
# ==========================

if __name__ == "__main__":
    print("📥 Reading DOI column from CSV...")
    dois = read_dois_from_csv(DOI_CSV, colname=DOI_COLNAME)
    print(f"Total {len(dois)} DOIs read.\n")

    print("🚀 Starting batch download of Wiley PDFs...")
    batch_download(dois, out_dir=OUTPUT_DIR)

    print("\n🎉 All tasks completed! PDFs saved in:", OUTPUT_DIR)