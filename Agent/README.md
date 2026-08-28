# BEAVER Agent Platform

BEAVER is a Streamlit-based multi-agent platform for evidence-grounded question answering and property-constrained design of degradable polymers.

## Installation

The validated development environment uses Python 3.12.3. Create and activate a suitable environment, then install the Python dependencies:

```bash
conda create -n beaver python=3.12
conda activate beaver
pip install -r requirements.txt
```

Pandoc must be installed separately for document conversion. The current Windows configuration expects `pandoc.exe` at `C:\Program Files\Pandoc\pandoc.exe`. Optional PDF conversion through `docx2pdf` requires Microsoft Word on Windows or macOS.

## Launch the web application

Open a terminal in the `APP` directory:

```bash
cd Agent/APP
streamlit run app.py
```

Open the local address shown by Streamlit, normally:

```text
http://localhost:8501
```

For remote deployment, run the same Streamlit application on a server and expose port `8501` through an authorized reverse proxy or hosting service. Deployment credentials and tunnel tokens are not included in this repository.

## Optional online demonstration

After the Streamlit application is running on port `8501`, the maintainer can expose it temporarily through cpolar:

```bat
cd /d PATH\TO\cpolar
cpolar http 8501
```

Copy the HTTPS forwarding address displayed by cpolar and provide it to reviewers:

```text
https://your-demo-address.cpolar.cn
```

Reviewers only need to open this HTTPS address in a web browser. They do not need to install cpolar or know the deployment token. With a free cpolar tunnel, the public address may change after the tunnel is restarted. The host computer, Streamlit process, and cpolar tunnel must remain running while the demonstration is being accessed.

The cpolar authentication token must be configured privately by the maintainer and must not be placed in this README or the source-code repository. Test-account credentials, if required, should be provided separately through the journal submission system.

## Register and use the platform

1. Open the BEAVER web page.
2. Select **Register new account**, enter a username and password, and complete registration.
3. Return to the **Login** tab and sign in.
4. In the sidebar, enter an OpenAI-compatible **Base URL** and **Security Token**.
5. Click **Connect** and wait until the interface reports that the Agent engine is ready.
6. Enter a literature question or polymer-design request in the chat box.
7. Optional memory and reasoning controls can be enabled from the sidebar.

The first account created in a new local database is assigned the administrator role. The account database is generated locally and is not distributed with the source code.

Users must provide their own API endpoint and API key. API credentials are retained in the current Streamlit session and are not included in this repository. The endpoint must support the model names configured in `Agent_Config/agent_config.py`.

## External resources

The large literature embedding arrays (`.npy`) and their associated metadata
files are not included in this GitHub repository. They are archived on Zenodo:

https://doi.org/10.5281/zenodo.22135808

Download and extract the chunk-level and/or paragraph-level corpus archives
before launching the Agent. Set the extracted corpus location through the
`BEAVER_CORPUS_ROOT` environment variable.

Windows Command Prompt:

```bat
set "BEAVER_CORPUS_ROOT=C:\path\to\BEAVER_Corpus"
set "BEAVER_CORPUS_GRANULARITY=chunk"
```

Linux:

```bash
export BEAVER_CORPUS_ROOT="/path/to/BEAVER_Corpus"
export BEAVER_CORPUS_GRANULARITY="chunk"
```

Set `BEAVER_CORPUS_GRANULARITY` to `paras` to use the paragraph-level corpus.

The BM25 index and other retrieval caches are generated automatically during
the first run and are therefore not included in the Zenodo archive.

## Folder structure

| Folder | Purpose |
|---|---|
| `Agent_Config/` | Stores model names, corpus locations, output settings, and the shared LLM/API client. |
| `APP/` | Contains the Streamlit entry point, login interface, sidebar configuration, chat interface, and web assets. |
| `Intent/` | Classifies user requests and routes them to question-answering or polymer-design workflows. |
| `memory/` | Implements short-term and long-term Agent memory. |
| `Planner/` | Decomposes requests, executes workflow steps, queries property data, and coordinates design tasks. |
| `Quality_loop/` | Performs answer-quality checks and knowledge-graph-assisted consistency inspection. |
| `RAG/` | Loads retrieval indexes and performs Dense, BM25, hybrid retrieval, reranking, and evidence-grounded answering. |
| `Report/` | Generates reports, Word documents, design summaries, and repeat-unit figures. |
| `resources/` | Contains the polymer-name keyword files and polymer-name/SMILES dictionary used for repeat-unit figures. |
| `Utils/` | Provides shared file, text, path, plotting, and document-conversion utilities. |

## Security

Do not place real API keys, service credentials, tunnel tokens, or populated user databases in the source-code package. Reviewers should use their own API credentials or access a separately hosted demonstration instance.

## Data availability

The large literature embeddings and associated metadata required by the
BEAVER retrieval workflow are available from Zenodo:

https://doi.org/10.5281/zenodo.22135808

The source code and configuration instructions are provided in this GitHub
repository.
