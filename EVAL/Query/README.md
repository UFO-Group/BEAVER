# Query Mode Evaluation

This folder contains the Query Mode evaluation used to compare BEAVER with raw and constrained LLM baselines. The benchmark contains 111 questions covering single, single-with-facets, and multi-part query structures.

## Main files

| Path | Purpose |
|---|---|
| `Query_script/single_singlewithfacets_multipart_dataset.json` | Query benchmark and ground-truth answers. |
| `Query_script/Query_BEAVER_VS_Raw_VS_Constr_forward.py` | Runs the forward-order evaluation. |
| `Query_script/Query_BEAVER_VS_Raw_VS_Constr_reverse.py` | Runs the reverse-order evaluation. |
| `Query_script/Query_draw.ipynb` | Summarizes scores and generates figures. |
| `Judge_output/` | Stores the evaluation scores, logs, and figures. |
| `Query_representative_output/` | Contains representative Query Mode outputs. |

## Judge API configuration

The forward- and reverse-order evaluation scripts require an OpenAI-compatible judge API. Set the following environment variables before running the scripts. Do not write a real API key into the source code or commit it to the repository.

Linux:

```bash
export JUDGE_API_KEY="your-api-key"
export JUDGE_BASE_URL="your-api-url"
export JUDGE_MODEL="your-judge-model"
```

Windows PowerShell:

```powershell
$env:JUDGE_API_KEY="your-api-key"
$env:JUDGE_BASE_URL="your-api-url"
$env:JUDGE_MODEL="your-judge-model"
```

These variables apply to the current terminal session and must be set again in a new terminal.

## Run

```bash
cd Query_script
python Query_BEAVER_VS_Raw_VS_Constr_forward.py
python Query_BEAVER_VS_Raw_VS_Constr_reverse.py
```

API access and the external RAG corpus must be configured before regenerating BEAVER answers. Do not commit real API keys to the repository.
