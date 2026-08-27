# Automated Literature Data Extraction

This folder contains the literature data-extraction workflow used to convert polymer-related PDF articles into a standardized CSV dataset. The complete workflow is controlled by `pipeline_main.py`.

## Workflow

1. Place the original PDF files in `Data/PDFs/`.
2. `Split_pdf.py` divides each PDF into five-page sections and saves them in `Data/Data_split/`.
3. `Data_Extraction/main_PDF_Youngsmodulus.py` filters the split PDFs by keywords and calls the large language model to extract polymer compositions and properties.
4. `Table_Generation.py` converts the extracted results into a structured CSV table.
5. `Standardize_Units.py` standardizes numerical values and physical units.
6. `Name_Classification.py` classifies polymer and material names using the dictionaries in `dict/`.
7. Final tables are saved in `Data/Processed_Results/`.

## Main scripts

| Script | Purpose |
|---|---|
| `pipeline_main.py` | Runs PDF splitting, LLM extraction, table generation, unit standardization, and name classification in sequence. |
| `Split_pdf.py` | Splits source PDFs into five-page PDF files. |
| `Data_Extraction/main_PDF_Youngsmodulus.py` | Extracts text, filters relevant PDF sections, and sends selected content to the LLM. |
| `Data_Extraction/API_YoungsModulus.py` | Provides the LLM API call used for structured property extraction. |
| `Data_Extraction/contains_keywords_youngs.py` | Detects polymer and property keywords in extracted text. |
| `Data_Extraction/Clean.py` | Cleans paragraphs and removes invalid or irrelevant text. |
| `Data_Extraction/TextNormalizer.py` | Normalizes characters, spaces, punctuation, and text formatting. |
| `Table_Generation.py` | Collects LLM outputs and generates a structured CSV table. |
| `Standardize_Units.py` | Standardizes mechanical-property and temperature units. |
| `Name_Classification.py` | Classifies polymer and material names using the dictionaries in `dict/`. |
| `Download/Doi_Abstract_Youngs_search.py` | Searches literature metadata and DOI records using property-related keywords. |
| `Download/PDFs_download.py` | Downloads PDF files from a DOI list when access is available. |

## API configuration

`Data_Extraction/API_YoungsModulus.py` reads the LLM API key and endpoint from environment variables. Set them before running the pipeline. Do not write a real API key into the source code, README, or other files committed with the project.

Linux:

```bash
export LLM_API_KEY="your-new-api-key"
export LLM_BASE_URL="your-url"
export LLM_MODEL="DeepSeek-R1"
```

Windows Command Prompt:

```bat
set "LLM_API_KEY=your-new-api-key"
set "LLM_BASE_URL=your-url"
set "LLM_MODEL=DeepSeek-R1"
```


These variables apply to the current terminal session. Set them again after opening a new terminal.

## Run

```bash
pip install -r requirements.txt
python pipeline_main.py
```

The main input, intermediate, and output paths are configured in `pipeline_config.json`. Configure the API environment variables above before starting the pipeline.

## Python version

```text
Python 3.12.3
```
