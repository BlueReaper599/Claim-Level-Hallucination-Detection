import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from hybrid_retriever import (
    build_models,
    hybrid_search,
)

from nli_verifier import NLIVerifier


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

EVAL_FILE = (
    RAW_DATA_DIR
    / "shared_task_dev.jsonl"
)

EVAL_LIMIT = 1000

NLI_TOP_K = 5

NLI_BATCH_SIZE = 16

DECISION_THRESHOLD = 0.50

DECISION_MARGIN = 0.10

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "UNKNOWN",
}


# ======================================================================
# DATA LOADING
# ======================================================================

def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    return records


# ======================================================================
# V1 AGGREGATION
# ======================================================================

def aggregate_v1(
    nli_results,
):

    max_entailment = max(
        result["probabilities"]["entailment"]
        for result in nli_results
    )

    max_contradiction = max(
        result["probabilities"]["contradiction"]
        for result in nli_results
    )

    max_neutral = max(
        result["probabilities"]["neutral"]
        for result in nli_results
    )

    if (
        max_contradiction
        > max_entailment
    ):

        label = "CONTRADICTED"
        confidence = max_contradiction

    elif (
        max_entailment
        > max_contradiction
    ):

        label = "SUPPORTED"
        confidence = max_entailment

    else:

        label = "UNKNOWN"
        confidence = max_neutral

    return label, confidence


# ======================================================================
# V2 AGGREGATION
# ======================================================================

def aggregate_v2(
    nli_results,
):

    weighted_entailment = 0.0
    weighted_contradiction = 0.0
    weighted_neutral = 0.0

    total_weight = 0.0

    for rank, result in enumerate(
        nli_results,
        start=1,
    ):

        weight = 1.0 / rank

        probabilities = (
            result["probabilities"]
        )

        weighted_entailment += (
            weight
            * probabilities["entailment"]
        )

        weighted_contradiction += (
            weight
            * probabilities["contradiction"]
        )

        weighted_neutral += (
            weight
            * probabilities["neutral"]
        )

        total_weight += weight

    entailment = (
        weighted_entailment
        / total_weight
    )

    contradiction = (
        weighted_contradiction
        / total_weight
    )

    neutral = (
        weighted_neutral
        / total_weight
    )

    scores = {
        "SUPPORTED": entailment,
        "CONTRADICTED": contradiction,
        "UNKNOWN": neutral,
    }

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_label = sorted_scores[0][0]

    best_score = sorted_scores[0][1]

    second_score = sorted_scores[1][1]

    margin = (
        best_score
        - second_score
    )

    if best_label == "UNKNOWN":

        final_label = "UNKNOWN"

    elif (
        best_score
        >= DECISION_THRESHOLD
        and
        margin
        >= DECISION_MARGIN
    ):

        final_label = best_label

    else:

        final_label = "UNKNOWN"

    return final_label, best_score


# ======================================================================
# TOP-1
# ======================================================================

def classify_top1(
    nli_result,
):

    probabilities = (
        nli_result["probabilities"]
    )

    scores = {
        "SUPPORTED":
            probabilities["entailment"],

        "CONTRADICTED":
            probabilities["contradiction"],

        "UNKNOWN":
            probabilities["neutral"],
    }

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_label = sorted_scores[0][0]

    best_score = sorted_scores[0][1]

    second_score = sorted_scores[1][1]

    margin = (
        best_score
        - second_score
    )

    if best_label == "UNKNOWN":

        final_label = "UNKNOWN"

    elif (
        best_score
        >= DECISION_THRESHOLD
        and
        margin
        >= DECISION_MARGIN
    ):

        final_label = best_label

    else:

        final_label = "UNKNOWN"

    return final_label, best_score


# ======================================================================
# EVALUATE ONE SYSTEM
# ======================================================================

def evaluate_system(
    system_name,
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[
            "SUPPORTED",
            "CONTRADICTED",
            "UNKNOWN",
        ],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "SUPPORTED",
            "CONTRADICTED",
            "UNKNOWN",
        ],
    )

    print("\n" + "=" * 80)

    print(
        f"{system_name}"
    )

    print("=" * 80)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {precision_macro:.4f}"
    )

    print(
        f"Macro Recall   : {recall_macro:.4f}"
    )

    print(
        f"Macro F1       : {f1_macro:.4f}"
    )

    print(
        f"Weighted F1    : {f1_weighted:.4f}"
    )

    print("\nClassification report:")

    print(report)

    print(
        "Confusion matrix:"
    )

    print(
        matrix
    )

    return {
        "system": system_name,
        "accuracy": float(accuracy),
        "macro_precision": float(
            precision_macro
        ),
        "macro_recall": float(
            recall_macro
        ),
        "macro_f1": float(
            f1_macro
        ),
        "weighted_f1": float(
            f1_weighted
        ),
        "confusion_matrix": matrix.tolist(),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 80)

    print(
        "TOP-1 VS TOP-5 NLI ABLATION"
    )

    print("=" * 80)

    print(
        f"Evaluation file: {EVAL_FILE}"
    )

    print(
        f"Evaluation limit: {EVAL_LIMIT}"
    )

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    records = load_jsonl(
        EVAL_FILE
    )

    records = records[
        :EVAL_LIMIT
    ]

    print(
        f"Loaded records: {len(records)}"
    )

    # --------------------------------------------------------------
    # Build retrieval models
    # --------------------------------------------------------------

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = build_models()

    # --------------------------------------------------------------
    # Load NLI
    # --------------------------------------------------------------

    nli_verifier = NLIVerifier(
        batch_size=NLI_BATCH_SIZE
    )

    # --------------------------------------------------------------
    # Storage
    # --------------------------------------------------------------

    y_true = []

    predictions_top1 = []

    predictions_v1 = []

    predictions_v2 = []

    detailed_results = []

    # --------------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------------

    for record in tqdm(
        records,
        desc="Evaluating",
    ):

        claim = record["claim"]

        gold_label = LABEL_MAP[
            record["label"]
        ]

        retrieved = hybrid_search(
            claim,
            semantic_index,
            bm25,
            semantic_metadata,
            bm25_metadata,
            embedding_model,
        )

        top5 = retrieved[
            :NLI_TOP_K
        ]

        evidence_texts = [
            item["text"]
            for item in top5
        ]

        nli_results = (
            nli_verifier.predict_batch(
                claim,
                evidence_texts,
            )
        )

        # ----------------------------------------------------------
        # TOP-1
        # ----------------------------------------------------------

        top1_label, top1_confidence = (
            classify_top1(
                nli_results[0]
            )
        )

        # ----------------------------------------------------------
        # V1
        # ----------------------------------------------------------

        v1_label, v1_confidence = (
            aggregate_v1(
                nli_results
            )
        )

        # ----------------------------------------------------------
        # V2
        # ----------------------------------------------------------

        v2_label, v2_confidence = (
            aggregate_v2(
                nli_results
            )
        )

        y_true.append(
            gold_label
        )

        predictions_top1.append(
            top1_label
        )

        predictions_v1.append(
            v1_label
        )

        predictions_v2.append(
            v2_label
        )

        detailed_results.append(
            {
                "id": record["id"],
                "claim": claim,
                "gold_label": gold_label,

                "top1": {
                    "label": top1_label,
                    "confidence":
                        top1_confidence,
                },

                "v1": {
                    "label": v1_label,
                    "confidence":
                        v1_confidence,
                },

                "v2": {
                    "label": v2_label,
                    "confidence":
                        v2_confidence,
                },

                "evidence": [
                    {
                        "page_id":
                            item["page_id"],

                        "sentence_id":
                            item["sentence_id"],

                        "text":
                            item["text"],

                        "retrieval_score":
                            item["score"],

                        "nli":
                            nli,
                    }

                    for item, nli in zip(
                        top5,
                        nli_results,
                    )
                ],
            }
        )

    # --------------------------------------------------------------
    # Evaluate systems
    # --------------------------------------------------------------

    results = []

    results.append(
        evaluate_system(
            "HYBRID + TOP-1 NLI",
            y_true,
            predictions_top1,
        )
    )

    results.append(
        evaluate_system(
            "HYBRID + TOP-5 V1",
            y_true,
            predictions_v1,
        )
    )

    results.append(
        evaluate_system(
            "HYBRID + TOP-5 V2",
            y_true,
            predictions_v2,
        )
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "SUMMARY"
    )

    print("=" * 80)

    print(
        f"{'System':<25}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Weighted F1':>14}"
    )

    print("-" * 63)

    for result in results:

        print(
            f"{result['system']:<25}"
            f"{result['accuracy']:>12.4f}"
            f"{result['macro_f1']:>12.4f}"
            f"{result['weighted_f1']:>14.4f}"
        )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    summary_file = (
        RESULTS_DIR
        / "top1_vs_top5_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    detailed_file = (
        RESULTS_DIR
        / "top1_vs_top5_detailed.jsonl"
    )

    with open(
        detailed_file,
        "w",
        encoding="utf-8",
    ) as f:

        for result in detailed_results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nResults saved to:")

    print(
        summary_file
    )

    print(
        detailed_file
    )


if __name__ == "__main__":

    main()