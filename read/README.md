# Publisher HTML Reading and Paragraph Extraction

This folder contains the HTML-processing workflow used to extract article metadata and body paragraphs from publisher webpages, select property-related text, and generate LLM-assisted labels and dense embeddings.

Supported publisher parsers include ACS, AIP, Elsevier, IOP, RSC, Springer, and Wiley.

## Workflow

1. Place publisher HTML files in `Data/<Publisher>/source/`.
2. Run the corresponding `0.*_read.ipynb` notebook to convert the HTML files into structured text files under `Data/<Publisher>/txt/`.
3. Run `1.HTML_Conversion_and_Heuristic_Paragraph_Filtering.ipynb` to select abstracts and paragraphs containing relevant polymer and property keywords.
4. Run `2.Paragraph_Extraction_and_Embedding.ipynb` to clean the selected paragraphs, generate LLM-assisted semantic labels, and create dense embedding outputs.

The current cells in step 3 are configured for RSC and Wiley data. The current cells in step 4 are configured for Wiley data. To process another publisher, change only the publisher name in the corresponding `DATA_ROOT / "<Publisher>" / ...` path definitions.

## Main files

| File or folder | Purpose |
|---|---|
| `0.ACS_read.ipynb` | Extracts metadata and paragraphs from ACS HTML files. The first code cell performs HTML extraction; the separate CSV-filtering cell is an optional utility and is not required by the main workflow. |
| `0.AIP_read.ipynb` | Extracts metadata and paragraphs from AIP HTML files. |
| `0.Elsevier_read.ipynb` | Extracts metadata and paragraphs from Elsevier HTML files. |
| `0.IOP_read.ipynb` | Extracts metadata and paragraphs from IOP HTML files. |
| `0.RSC_read.ipynb` | Extracts metadata and paragraphs from RSC HTML files. |
| `0.Springer_read.ipynb` | Extracts metadata and paragraphs from Springer HTML files. |
| `0.wiley_read.ipynb` | Extracts metadata and paragraphs from Wiley HTML files. |
| `1.HTML_Conversion_and_Heuristic_Paragraph_Filtering.ipynb` | Selects abstracts and paragraphs using polymer and property keywords. |
| `2.Paragraph_Extraction_and_Embedding.ipynb` | Cleans selected paragraphs and generates semantic labels, metadata tables, and dense embeddings. |
| `read/*.py` | Implements publisher-specific parsers and shared text, paragraph, and table-processing functions. |
| `read/*_read_try.ipynb` | Provides small publisher-specific parser examples using `Data/examples/<Publisher>/example.html`. |

## Data layout

All paths are relative to this folder. The input data directories do not need to be included with the source-code package; create only the directories required for the publisher being processed.

```text
Data/
  <Publisher>/
    source/          # Input HTML files
    txt/             # Extracted metadata and paragraphs
    hit_abstracts/   # Keyword-matched abstracts, when enabled
    hit_paragraphs/  # Keyword-matched paragraphs
    plain_text/      # Clean paragraph text
    embeddings/      # Labels, metadata, and dense embeddings
  examples/
    <Publisher>/
      example.html
```

Run Jupyter from this top-level `read` folder so that `DATA_ROOT = Path("Data")` resolves correctly.

## Environment setup

The notebooks were prepared with Python 3.12.3.

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

After Jupyter opens, run the required notebooks in the workflow order shown above.

## LLM configuration

Only `2.Paragraph_Extraction_and_Embedding.ipynb` requires an OpenAI-compatible API. Configure the credentials before starting Jupyter.

Windows Command Prompt:

```bat
set "LLM_API_KEY=your-api-key"
set "LLM_BASE_URL=https://your-compatible-api.example/v1"
set "LLM_MODEL=your-chat-model"
jupyter notebook
```

Linux:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://your-compatible-api.example/v1"
export LLM_MODEL="your-chat-model"
jupyter notebook
```

`LLM_API_KEY` is required. `LLM_BASE_URL` and `LLM_MODEL` have defaults in the notebook but may be overridden. The embedding request currently uses the model name `GLM-Embedding-2`; the configured endpoint must provide that model, or the model name must be changed to one supported by the selected provider.

Never store real API keys in source files, notebooks, README files, screenshots, or committed shell scripts.

## Data and sharing notes

Publisher HTML files may be subject to third-party copyright and licence restrictions and are therefore not included in this source-code folder. Users must obtain the source documents through lawful access routes and comply with the relevant publisher terms.

Before sharing or archiving the project:

- Clear all Notebook outputs and execution counts.
- Delete `.ipynb_checkpoints/` and `__pycache__/` directories.
- Confirm that no local absolute paths, API keys, account information, webpage-session metadata, or downloaded full-text HTML files are included.

