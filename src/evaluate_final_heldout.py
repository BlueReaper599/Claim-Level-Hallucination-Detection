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

from hybrid_retriever import build_models, hybrid_search
from nli_verifier import NLIVerifier


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE = RAW_DATA_DIR / "shared_task_dev.jsonl"

# ------------------------------------------------------------
# IMPORTANT:
# The first 1000 records were used for architecture ablations.
# We therefore start the held-out evaluation AFTER record 1000.
# ------------------------------------------------------------

DEV_ABLATION_SIZE = 1000

HELDOUT_SIZE = 2000

HELDOUT_START = DEV_ABLATION_SIZE
HELDOUT_END = HELDOUT_START + HELDOUT_SIZE

DECISION_THRESHOLD = 0.50
DECISION_MARGIN = 0.10

NLI_BATCH_SIZE = 16

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "UNKNOWN",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


# ============================================================
# TOP-1 NLI DECISION
# ============================================================

def classify_nli_result(nli_result):
    """
    Convert the NLI probabilities into the project's
    three-way factuality classification.

    entailment     -> SUPPORTED
    contradiction  -> CONTRADICTED
    neutral        -> UNKNOWN

    A confidence threshold and margin are retained from
    the development experiments.
    """

    probabilities = nli_result["probabilities"]

    scores = {
        "SUPPORTED": probabilities["entailment"],
        "CONTRADICTED": probabilities["contradiction"],
        "UNKNOWN": probabilities["neutral"],
    }

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_label = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    second_score = sorted_scores[1][1]

    margin = best_score - second_score

    if best_label == "UNKNOWN":
        final_label = "UNKNOWN"

    elif (
        best_score >= DECISION_THRESHOLD
        and margin >= DECISION_MARGIN
    ):
        final_label = best_label

    else:
        final_label = "UNKNOWN"

    return final_label, best_score, margin


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    labels = [
        "SUPPORTED",
        "CONTRADICTED",
        "UNKNOWN",
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        precision_macro,
        recall_macro,
        f1_macro,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
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

        "weighted_precision": float(
            precision_weighted
        ),

        "weighted_recall": float(
            recall_weighted
        ),

        "weighted_f1": float(
            f1_weighted
        ),

        "classification_report": report_dict,

        "confusion_matrix": matrix.tolist(),

        "classification_report_text": report_text,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL HELD-OUT EVALUATION")
    print("=" * 80)

    print(f"Evaluation file : {EVAL_FILE}")
    print(
        f"Development/ablation records : "
        f"0 - {DEV_ABLATION_SIZE - 1}"
    )

    print(
        f"Held-out records : "
        f"{HELDOUT_START} - {HELDOUT_END - 1}"
    )

    print(
        f"Held-out size : {HELDOUT_SIZE}"
    )

    # --------------------------------------------------------
    # Load complete FEVER dev file
    # --------------------------------------------------------

    all_records = load_jsonl(EVAL_FILE)

    print(
        f"\nTotal records available: "
        f"{len(all_records)}"
    )

    if HELDOUT_END > len(all_records):

        raise ValueError(
            "Requested held-out range exceeds "
            "the available dataset."
        )

    # --------------------------------------------------------
    # Select held-out records
    # --------------------------------------------------------

    records = all_records[
        HELDOUT_START:HELDOUT_END
    ]

    print(
        f"Held-out records loaded: "
        f"{len(records)}"
    )

    if not records:
        raise ValueError(
            "No held-out records were selected."
        )

    # --------------------------------------------------------
    # Load retrieval system
    # --------------------------------------------------------

    print("\nLoading retrieval system...")

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = build_models()

    # --------------------------------------------------------
    # Load NLI verifier
    # --------------------------------------------------------

    print("\nLoading NLI verifier...")

    nli_verifier = NLIVerifier(
        batch_size=NLI_BATCH_SIZE
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    y_true = []
    y_pred = []

    detailed_results = []

    # --------------------------------------------------------
    # Evaluate held-out examples
    # --------------------------------------------------------

    for record in tqdm(
        records,
        desc="Held-out evaluation",
    ):

        claim = record["claim"]

        gold_label = LABEL_MAP[
            record["label"]
        ]

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        retrieved = hybrid_search(
            claim,
            semantic_index,
            bm25,
            semantic_metadata,
            bm25_metadata,
            embedding_model,
        )

        if not retrieved:
            predicted_label = "UNKNOWN"
            confidence = 0.0
            margin = 0.0

            top_evidence = None
            nli_result = None

        else:

            # ------------------------------------------------
            # Frozen architecture:
            #
            # Hybrid retrieval
            #       ↓
            # Top-1 evidence
            #       ↓
            # NLI
            # ------------------------------------------------

            top_evidence = retrieved[0]

            nli_result = nli_verifier.predict(
                claim,
                top_evidence["text"],
            )

            (
                predicted_label,
                confidence,
                margin,
            ) = classify_nli_result(
                nli_result
            )

        y_true.append(gold_label)
        y_pred.append(predicted_label)

        # ----------------------------------------------------
        # Save detailed prediction
        # ----------------------------------------------------

        detailed_results.append({
            "dataset_index": (
                HELDOUT_START
                + len(detailed_results)
            ),

            "id": record["id"],

            "claim": claim,

            "gold_label": gold_label,

            "predicted_label": predicted_label,

            "confidence": confidence,

            "margin": margin,

            "retrieval": {
                "method": "HYBRID_BM25_MINILM_RRF",
                "top_k_for_search": 10,
            },

            "top_evidence": (
                {
                    "page_id": top_evidence[
                        "page_id"
                    ],

                    "sentence_id": top_evidence[
                        "sentence_id"
                    ],

                    "text": top_evidence[
                        "text"
                    ],

                    "retrieval_score": top_evidence[
                        "score"
                    ],
                }
                if top_evidence is not None
                else None
            ),

            "nli": nli_result,
        })

    # ========================================================
    # Calculate final metrics
    # ========================================================

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    # ========================================================
    # Print results
    # ========================================================

    print("\n" + "=" * 80)
    print("HELD-OUT RESULTS")
    print("=" * 80)

    print(
        f"Examples evaluated : {len(records)}"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted Precision  : "
        f"{metrics['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall     : "
        f"{metrics['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print("\nClassification report:")

    print(
        metrics[
            "classification_report_text"
        ]
    )

    print("Confusion matrix:")

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    # ========================================================
    # Dataset label distribution
    # ========================================================

    print("\nHeld-out gold label distribution:")

    for label in [
        "SUPPORTED",
        "CONTRADICTED",
        "UNKNOWN",
    ]:

        count = y_true.count(label)

        percentage = (
            100.0 * count / len(y_true)
        )

        print(
            f"{label:<15}"
            f"{count:>6}"
            f" ({percentage:>6.2f}%)"
        )

    # ========================================================
    # Prediction distribution
    # ========================================================

    print("\nHeld-out prediction distribution:")

    for label in [
        "SUPPORTED",
        "CONTRADICTED",
        "UNKNOWN",
    ]:

        count = y_pred.count(label)

        percentage = (
            100.0 * count / len(y_pred)
        )

        print(
            f"{label:<15}"
            f"{count:>6}"
            f" ({percentage:>6.2f}%)"
        )

    # ========================================================
    # Save summary
    # ========================================================

    summary = {
        "experiment": "final_heldout_evaluation",

        "dataset": {
            "file": str(EVAL_FILE),
            "total_records_available": len(
                all_records
            ),

            "development_ablation_range": [
                0,
                DEV_ABLATION_SIZE - 1,
            ],

            "heldout_range": [
                HELDOUT_START,
                HELDOUT_END - 1,
            ],

            "heldout_size": len(records),
        },

        "architecture": {
            "retrieval": (
                "BM25 + MiniLM semantic retrieval "
                "with Reciprocal Rank Fusion"
            ),

            "evidence_selection": (
                "Top-1 retrieved sentence"
            ),

            "nli_model": (
                "cross-encoder/nli-deberta-v3-base"
            ),

            "decision_threshold": (
                DECISION_THRESHOLD
            ),

            "decision_margin": (
                DECISION_MARGIN
            ),
        },

        "metrics": metrics,

        "gold_label_distribution": {
            label: y_true.count(label)
            for label in [
                "SUPPORTED",
                "CONTRADICTED",
                "UNKNOWN",
            ]
        },

        "prediction_distribution": {
            label: y_pred.count(label)
            for label in [
                "SUPPORTED",
                "CONTRADICTED",
                "UNKNOWN",
            ]
        },
    }

    summary_file = (
        RESULTS_DIR /
        "final_heldout_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Save detailed predictions
    # ========================================================

    detailed_file = (
        RESULTS_DIR /
        "final_heldout_detailed.jsonl"
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

    # ========================================================
    # Finish
    # ========================================================

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(summary_file)
    print(detailed_file)

    print("\nHeld-out evaluation complete.")


if __name__ == "__main__":
    main()