# BEAVER

BEAVER is an evidence-grounded multi-agent framework for scientific question answering and property-constrained design of degradable polymers. This repository contains the source code, data-processing workflows, and evaluation files used in the study.

## Repository structure

| Folder | Purpose |
|---|---|
| [`read/`](read/) | Processes literature HTML files and constructs retrieval corpora. |
| [`ML_Tool/`](ML_Tool/) | Extracts polymer-property data and performs machine-learning training and prediction. |
| [`Agent/`](Agent/) | Contains the BEAVER multi-agent system and Streamlit web application. |
| [`EVAL/`](EVAL/) | Contains the Query, RAG, and Design evaluation scripts and results. |

Installation, configuration, and execution instructions are provided in the README for each component:

- [Literature processing](read/README.md)
- [Machine-learning tools](ML_Tool/README.md)
- [BEAVER Agent](Agent/README.md)
- [Evaluation](EVAL/README.md)

## Data availability

Large literature embeddings, metadata, machine-learning grid-search outputs, and final cross-validation metrics are archived on Zenodo:

https://doi.org/10.5281/zenodo.22135808

The required directory layouts and configuration steps are described in the corresponding component READMEs. Publisher PDF files and licensed full-text source documents are not redistributed in this repository.

## License

Original BEAVER source code is released under the [MIT License](LICENSE). Third-party software, publisher content, and external datasets remain subject to their respective licenses and terms of use.
