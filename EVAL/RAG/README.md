# RAG Evaluation

This folder contains the retrieval-augmented generation (RAG) evaluation used in the manuscript. The benchmark contains 111 question–ground-truth pairs and compares paragraph- and chunk-based retrieval.

## Evaluation settings

- Retrieval methods: Dense, BM25, and Hybrid
- Retrieval conditions: Raw and LLM reranking
- Retrieved candidate pool: 45
- Final retained documents: 15
- Metrics: Cosine Similarity, BERTScore, and RAGAS

## Folder structure

| Path | Purpose |
|---|---|
| `para/` | Paragraph-based evaluation results and figures. |
| `chunk/` | Chunk-based evaluation results and figures. |
| `RAG_script/test_dataset.json` | The 111-question RAG benchmark. |
| `RAG_script/RAG_BERTScore_COS.py` | Generates answers and calculates Cosine Similarity and BERTScore. |
| `RAG_script/RAG_AS.py` | Generates answers and evaluates them using RAGAS. |
| `RAG_script/RAG_BERTScore_COS_draw.ipynb` | Draws the Cosine Similarity and BERTScore figures. |
| `RAG_script/RAG_AS_draw_completed.ipynb` | Draws the RAGAS figures. |

Each retrieval-method folder contains the question-level results, metric details, summary tables, and distribution figures. The final manuscript figures are stored directly under `para/BERT_COS`, `para/RAG_AS`, `chunk/BERT_COS`, and `chunk/RAG_AS`.

## API configuration

The evaluation scripts require an OpenAI-compatible API endpoint. Set the API key and base URL as environment variables before running the scripts. Do not write a real API key into the source code or commit it to the repository.

Linux:

```bash
export RAG_API_KEY="your_api_key"
export RAG_BASE_URL="your_api_url"
```

Windows PowerShell:

```powershell
$env:RAG_API_KEY="your_api_key"
$env:RAG_BASE_URL="your_api_url"
```

These variables apply to the current terminal session and must be set again in a new terminal.

## Run

Run the evaluation scripts from `RAG_script`:

```bash
cd RAG_script
python RAG_BERTScore_COS.py
python RAG_AS.py
```

Select Dense, BM25, or Hybrid when prompted. Run the drawing notebooks from the same directory so that their relative paths resolve correctly.

Run the evaluation once with the chunk corpus configuration and once with the paragraph corpus configuration. The active corpus is selected through `CORPUS_CONFIG` in `Agent/Agent_Config/agent_config.py`; the embedding paths, metadata paths, and `granularity` values must all refer to the same corpus level. The scripts automatically save results under `chunk/` or `para/` according to that configuration. The drawing notebooks contain separate plotting cells for the paragraph- and chunk-based result folders rather than a single all-in-one plotting command.

The scripts import the RAG implementation from the repository-level `Agent` folder. BM25 caches, dense embeddings, metadata files, and API access must be configured separately before regenerating the results. Do not commit real API keys to the repository.
