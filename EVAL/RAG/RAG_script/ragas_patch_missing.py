# -*- coding: utf-8 -*-
"""Recompute only the nine missing RAGAS cells without touching source files.

Place this script next to RAG_AS.py in EVAL/RAG/RAG_script.  It reads the
original para/chunk CSV files, evaluates only the missing metrics, and writes
a complete copied result tree to EVAL/RAG/RAGAS_completed.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from RAG_AS import (
    DeepSeekRagasEmbeddings,
    DeepSeekRagasLLM,
    LLM_MODEL_RAGAS,
    USER_API_KEY,
    USER_BASE_URL,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = SOURCE_ROOT / "RAGAS_completed"

GROUP_FOLDERS = [
    Path(level) / "RAG_AS" / mode
    for level in ("chunk", "para")
    for mode in ("Bm25", "Dense", "Hybrid")
]

CSV_FILES_TO_COPY = [
    "ragas_eval_dataset.csv",
    "ragas_raw_detail.csv",
    "ragas_rerank_detail.csv",
    "ragas_summary.csv",
]

TARGETS = [
    {
        "detail": Path("chunk/RAG_AS/Hybrid/ragas_raw_detail.csv"),
        "dataset": Path("chunk/RAG_AS/Hybrid/ragas_eval_dataset.csv"),
        "index": 6,
        "variant": "raw",
        "metrics": ["context_precision", "context_recall", "answer_correctness"],
    },
    {
        "detail": Path("para/RAG_AS/Bm25/ragas_rerank_detail.csv"),
        "dataset": Path("para/RAG_AS/Bm25/ragas_eval_dataset.csv"),
        "index": 31,
        "variant": "rerank",
        "metrics": ["faithfulness"],
    },
    {
        "detail": Path("para/RAG_AS/Dense/ragas_raw_detail.csv"),
        "dataset": Path("para/RAG_AS/Dense/ragas_eval_dataset.csv"),
        "index": 54,
        "variant": "raw",
        "metrics": ["faithfulness", "answer_correctness"],
    },
    {
        "detail": Path("para/RAG_AS/Hybrid/ragas_rerank_detail.csv"),
        "dataset": Path("para/RAG_AS/Hybrid/ragas_eval_dataset.csv"),
        "index": 15,
        "variant": "rerank",
        "metrics": ["faithfulness"],
    },
    {
        "detail": Path("para/RAG_AS/Hybrid/ragas_rerank_detail.csv"),
        "dataset": Path("para/RAG_AS/Hybrid/ragas_eval_dataset.csv"),
        "index": 54,
        "variant": "rerank",
        "metrics": ["faithfulness", "answer_correctness"],
    },
]

METRIC_FACTORIES = {
    "faithfulness": Faithfulness,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "answer_correctness": AnswerCorrectness,
}

SUMMARY_METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer Relevancy"),
    ("context_precision", "Context Precision"),
    ("context_recall", "Context Recall"),
    ("answer_correctness", "Answer Correctness"),
]


def is_missing(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a"}


def parse_contexts(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("Context is not a list or a serialized list.")


def make_sample(row: pd.Series, variant: str) -> Dataset:
    answer_col = "ans_raw" if variant == "raw" else "ans_rerank"
    context_col = "ctx_raw" if variant == "raw" else "ctx_rerank"
    frame = pd.DataFrame(
        [
            {
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "answer": row[answer_col],
                "contexts": parse_contexts(row[context_col]),
            }
        ]
    )
    return Dataset.from_pandas(frame, preserve_index=False)


def copy_source_csvs() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_ROOT}\n"
            "Rename or remove that completed-output directory before rerunning. "
            "The original para/chunk directories are never touched."
        )

    for relative_folder in GROUP_FOLDERS:
        source_folder = SOURCE_ROOT / relative_folder
        output_folder = OUTPUT_ROOT / relative_folder
        output_folder.mkdir(parents=True, exist_ok=False)
        for filename in CSV_FILES_TO_COPY:
            source_file = source_folder / filename
            if not source_file.exists():
                raise FileNotFoundError(f"Missing required source CSV: {source_file}")
            shutil.copy2(source_file, output_folder / filename)


def refresh_summary(folder: Path) -> None:
    raw = pd.read_csv(folder / "ragas_raw_detail.csv")
    rerank = pd.read_csv(folder / "ragas_rerank_detail.csv")
    rows = []
    for metric, label in SUMMARY_METRICS:
        raw_mean = pd.to_numeric(raw[metric], errors="coerce").mean()
        rerank_mean = pd.to_numeric(rerank[metric], errors="coerce").mean()
        improvement = "N/A" if raw_mean == 0 else f"{(rerank_mean - raw_mean) / raw_mean * 100:+.2f}%"
        rows.append(
            {
                "Metric": label,
                "Raw (Baseline)": raw_mean,
                "Rerank (Experiment)": rerank_mean,
                "Improvement": improvement,
            }
        )
    pd.DataFrame(rows).to_csv(
        folder / "ragas_summary.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    llm = DeepSeekRagasLLM(
        model=LLM_MODEL_RAGAS,
        openai_api_key=USER_API_KEY,
        openai_api_base=USER_BASE_URL,
    )
    embeddings = DeepSeekRagasEmbeddings()
    run_config = RunConfig(max_workers=1, timeout=240)

    source_frames: dict[Path, pd.DataFrame] = {}
    computed: list[tuple[Path, int, str, float]] = []

    # Complete every API calculation before creating or writing the output tree.
    for task_number, task in enumerate(TARGETS, start=1):
        source_detail = SOURCE_ROOT / task["detail"]
        source_dataset = SOURCE_ROOT / task["dataset"]
        if not source_detail.exists() or not source_dataset.exists():
            raise FileNotFoundError(
                f"Missing source input: {source_detail} or {source_dataset}"
            )

        detail_df = source_frames.setdefault(source_detail, pd.read_csv(source_detail))
        dataset_df = pd.read_csv(source_dataset)
        index = task["index"]
        if index >= len(detail_df) or index >= len(dataset_df):
            raise IndexError(f"Index {index} is outside {source_detail}")

        metrics_to_run = [
            metric
            for metric in task["metrics"]
            if is_missing(detail_df.at[index, metric])
        ]
        if not metrics_to_run:
            print(f"[{task_number}/5] Source row is already complete; skipping.")
            continue

        print(
            f"[{task_number}/5] {task['detail'].parent}, index={index}, "
            f"metrics={metrics_to_run}"
        )
        result = evaluate(
            make_sample(dataset_df.iloc[index], task["variant"]),
            metrics=[METRIC_FACTORIES[name]() for name in metrics_to_run],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=True,
        ).to_pandas()

        for metric in metrics_to_run:
            value = result.at[0, metric]
            if is_missing(value):
                raise RuntimeError(
                    f"RAGAS returned a missing value for {task['detail']}, "
                    f"index={index}, metric={metric}"
                )
            numeric_value = float(value)
            if not 0.0 <= numeric_value <= 1.0:
                raise ValueError(
                    f"Out-of-range result {numeric_value} for {metric}"
                )
            computed.append((task["detail"], index, metric, numeric_value))
            print(f"    {metric} = {numeric_value:.8f}")

    if not computed:
        print("The source target cells are already complete. No files were created.")
        return

    copy_source_csvs()

    output_frames: dict[Path, pd.DataFrame] = {}
    affected_folders = set()
    for relative_detail, index, metric, value in computed:
        output_detail = OUTPUT_ROOT / relative_detail
        frame = output_frames.setdefault(output_detail, pd.read_csv(output_detail))
        if not is_missing(frame.at[index, metric]):
            raise RuntimeError(
                f"Refusing to overwrite a non-missing copied value: "
                f"{output_detail}, index={index}, metric={metric}"
            )
        frame.at[index, metric] = value
        affected_folders.add(output_detail.parent)

    for output_detail, frame in output_frames.items():
        frame.to_csv(output_detail, index=False, encoding="utf-8-sig")
        print(f"Completed detail written: {output_detail}")

    for folder in sorted(affected_folders):
        refresh_summary(folder)
        print(f"Completed summary written: {folder / 'ragas_summary.csv'}")

    missing_after = []
    for task in TARGETS:
        frame = pd.read_csv(OUTPUT_ROOT / task["detail"])
        for metric in task["metrics"]:
            if is_missing(frame.at[task["index"], metric]):
                missing_after.append(
                    (task["detail"], task["index"], metric)
                )
    if missing_after:
        raise RuntimeError(f"Missing cells remain in completed copies: {missing_after}")

    print("Success: all nine missing cells were filled in copied files only.")
    print(f"Completed output root: {OUTPUT_ROOT}")
    print("Original para/chunk files were not modified.")


if __name__ == "__main__":
    main()
