#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.errors import PdfReadError

def split_pdf_by_chunk(input_pdf_path, output_folder, chunk_size=5):
    os.makedirs(output_folder, exist_ok=True)

    try:
        reader = PdfReader(input_pdf_path)
    except PdfReadError as e:
        print(f"❌ Unable to read PDF: {input_pdf_path}, Error message: {e}")
        return
    except Exception as e:
        print(f"❌ Unknown error reading PDF: {input_pdf_path}, Error message: {e}")
        return

    total_pages = len(reader.pages)
    base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]

    for start in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        end = min(start + chunk_size, total_pages)

        for i in range(start, end):
            writer.add_page(reader.pages[i])

        part_num = start // chunk_size + 1
        output_path = os.path.join(output_folder, f"{base_name}_part_{part_num}.pdf")

        with open(output_path, "wb") as f_out:
            writer.write(f_out)

        print(f"✅ Saved: {output_path}")

    print(f"🎉 Completed: Splitting of {input_pdf_path}, Total {total_pages} pages, {chunk_size} pages per chunk.")


def split_all_pdfs_in_folder(input_folder, output_base_folder, chunk_size=5):
    pdf_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(".pdf") and not f.startswith("._")]
    
    if not pdf_files:
        print(f"❌ No PDF files found in input directory: {input_folder}")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    for filename in pdf_files:
        input_pdf_path = os.path.join(input_folder, filename)
        output_folder = os.path.join(output_base_folder, os.path.splitext(filename)[0])

        try:
            split_pdf_by_chunk(input_pdf_path, output_folder, chunk_size)
        except Exception as e:
            print(f"⚠️ Skipping file: {filename}, Reason: {e}")


if __name__ == "__main__":
    # Read configuration
    import json
    try:
        with open("pipeline_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        input_folder = config.get("pdf_raw_dir", "Data")
        output_base_folder = config.get("pdf_split_dir", "Data/Data_split")
        chunk_size = config.get("chunk_size", 5)
    except:
        # Default configuration
        input_folder = "Data"
        output_base_folder = "Data/Data_split"
        chunk_size = 5
    
    print(f"🔧 Configuration:")
    print(f"  Input directory: {input_folder}")
    print(f"  Output directory: {output_base_folder}")
    print(f"  Chunk size: {chunk_size} pages")
    
    split_all_pdfs_in_folder(input_folder, output_base_folder, chunk_size)