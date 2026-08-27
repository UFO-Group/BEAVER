# BEAVER Evaluation

This folder contains the evaluation workflows and results used in the manuscript.

| Folder | Purpose |
|---|---|
| `RAG/` | Evaluates paragraph- and chunk-based retrieval using Cosine Similarity, BERTScore, and RAGAS. |
| `Query/` | Compares BEAVER Query Mode with raw and constrained LLM baselines. |
| `Design/` | Evaluates BEAVER and baseline models on coupled-property polymer design tasks using LLM judges. |

Each subfolder contains its evaluation scripts, input data, result files, figures, and a separate README describing the main files.

## Installation

```bash
pip install -r requirements.txt
```

Some evaluations also require the repository-level `Agent` folder, external BM25 and dense-retrieval resources, and access to the configured LLM APIs. Do not commit real API keys to the repository.
