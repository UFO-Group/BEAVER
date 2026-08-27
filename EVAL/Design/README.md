# Design Evaluation

This folder contains the polymer-design outputs and the scripts used to evaluate BEAVER and direct LLM baselines.

## Folder structure

| Folder | Description |
|---|---|
| `Mech._Deg._BEAVER_output/` | BEAVER outputs for mechanical and degradation constraints. |
| `Mech._Therm._BEAVER_output/` | BEAVER outputs for mechanical and thermal constraints. |
| `Mech._Therm._Deg._BEAVER_output/` | BEAVER outputs for mechanical, thermal, and degradation constraints. |
| `Therm._Deg._BEAVER_output/` | BEAVER outputs for thermal and degradation constraints. |
| `Judge_Input/` | Final BEAVER documents and direct-LLM outputs used as Judge inputs. |
| `Judge_output/` | Forward/reverse Judge scores, summary tables, and figures. |
| `Design_script/` | Baseline-generation, Judge-scoring, and plotting scripts. |

## Main scripts

| Script | Purpose |
|---|---|
| `Design-GPT55.ipynb` | Generates design responses with GPT-5.5. |
| `Design-DeepseekV4pro.ipynb` | Generates design responses with DeepSeek-V4-Pro. |
| `Design-Gemini31.ipynb` | Generates design responses with Gemini 3.1 Pro. |
| `Design-Qwen3.5.ipynb` | Generates design responses with Qwen 3.5 Plus. |
| `Design_BEAVER_vs_Models_claude.py` | Evaluates all design candidates with the Claude Judge. |
| `Design_BEAVER_vs_Models_gpt.py` | Evaluates all design candidates with the GPT Judge. |
| `Design_BEAVER_vs_Models_deepseek.py` | Evaluates all design candidates with the DeepSeek Judge. |
| `Design_Judge_draw.ipynb` | Summarizes Judge results and generates comparison figures. |

## Run a Judge

Set the API configuration before running a Judge script.

Linux:

```bash
export JUDGE_API_KEY="your-api-key"
export JUDGE_BASE_URL="your-api-url"
export JUDGE_MODEL="your-judge-model"
python Design_script/Design_BEAVER_vs_Models_claude.py
```

Windows PowerShell:

```powershell
$env:JUDGE_API_KEY="your-api-key"
$env:JUDGE_BASE_URL="your-api-url"
$env:JUDGE_MODEL="your-judge-model"
python Design_script/Design_BEAVER_vs_Models_claude.py
```

Use the corresponding GPT or DeepSeek script to reproduce the other Judge results. Inputs are read from `Judge_Input/`, and results are saved under `Judge_output/`.
