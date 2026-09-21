import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from hybrid_retriever import (
    build_models,
    hybrid_search,
)

from nli_verifier import (
    NLIVerifier,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "shared_task_dev.jsonl"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVAL_LIMIT = 1000

RETRIEVAL_TOP_K = 10

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
        encoding="utf-8"
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
    nli_results
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

    return {
        "label": label,
        "confidence": confidence,
        "max_entailment": max_entailment,
        "max_contradiction": max_contradiction,
        "max_neutral": max_neutral,
    }


# ======================================================================
# V2 AGGREGATION
# ======================================================================

def aggregate_v2(
    nli_results
):

    weighted_entailment = 0.0

    weighted_contradiction = 0.0

    weighted_neutral = 0.0

    total_weight = 0.0

    for rank, result in enumerate(
        nli_results,
        start=1
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
        reverse=True
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

    return {
        "label": final_label,
        "confidence": best_score,
        "margin": margin,
        "scores": scores,
    }


# ======================================================================
# RUN NLI ON RETRIEVED EVIDENCE
# ======================================================================

def verify_with_nli(
    claim,
    retrieved,
    nli_verifier
):

    evidence_items = (
        retrieved[:NLI_TOP_K]
    )

    evidence_texts = [
        item["text"]
        for item in evidence_items
    ]

    nli_results = (
        nli_verifier.predict_batch(
            claim,
            evidence_texts
        )
    )

    combined = []

    for item, nli_result in zip(
        evidence_items,
        nli_results
    ):

        combined.append(
            {
                "page_id": item.get(
                    "page_id"
                ),
                "sentence_id": item.get(
                    "sentence_id"
                ),
                "text": item["text"],
                "rrf_score": item.get(
                    "rrf_score"
                ),
                "nli": nli_result,
            }
        )

    return combined


# ======================================================================
# RETRIEVAL ADAPTERS
# ======================================================================

def retrieve_bm25(
    claim,
    bm25,
    metadata,
    top_k
):

    tokenized_query = claim.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        record = metadata[index]

        results.append(
            {
                "page_id": record.get(
                    "page_id"
                ),
                "sentence_id": record.get(
                    "sentence_id"
                ),
                "text": record["text"],
                "bm25_score": float(
                    scores[index]
                ),
            }
        )

    return results


def retrieve_semantic(
    claim,
    embedding_model,
    semantic_index,
    metadata,
    top_k
):

    query_embedding = (
        embedding_model.encode(
            [claim],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    )

    query_embedding = (
        query_embedding.astype(
            "float32"
        )
    )

    scores, indices = (
        semantic_index.search(
            query_embedding,
            top_k
        )
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        record = metadata[index]

        results.append(
            {
                "page_id": record.get(
                    "page_id"
                ),
                "sentence_id": record.get(
                    "sentence_id"
                ),
                "text": record["text"],
                "semantic_score": float(
                    score
                ),
            }
        )

    return results


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_system(
    system_name,
    records,
    retrieval_function,
    nli_verifier,
    aggregation_function,
    output_file
):

    print("\n")
    print("=" * 80)
    print(
        f"EVALUATING: {system_name}"
    )
    print("=" * 80)

    y_true = []

    y_pred = []

    detailed_predictions = []

    for i, record in enumerate(
        records,
        start=1
    ):

        claim = record["claim"]

        gold_label = LABEL_MAP[
            record["label"]
        ]

        retrieved = retrieval_function(
            claim
        )

        retrieved = retrieved[
            :RETRIEVAL_TOP_K
        ]

        nli_results = (
            verify_with_nli(
                claim,
                retrieved,
                nli_verifier
            )
        )

        aggregation = (
            aggregation_function(
                [
                    item["nli"]
                    for item in nli_results
                ]
            )
        )

        predicted_label = (
            aggregation["label"]
        )

        y_true.append(
            gold_label
        )

        y_pred.append(
            predicted_label
        )

        detailed_predictions.append(
            {
                "id": record.get(
                    "id"
                ),
                "claim": claim,
                "gold_label": gold_label,
                "predicted_label":
                    predicted_label,
                "aggregation":
                    aggregation,
                "evidence":
                    nli_results,
            }
        )

        if (
            i % 10 == 0
            or i == len(records)
        ):

            print(
                f"Processed "
                f"{i}/{len(records)}"
            )

    labels = [
        "SUPPORTED",
        "CONTRADICTED",
        "UNKNOWN",
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0
        )
    )

    print("\n")
    print("-" * 80)
    print(
        f"RESULTS: {system_name}"
    )
    print("-" * 80)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted Precision: "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall   : "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    print("Confusion matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print(
        "Labels:",
        labels
    )

    print(cm)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for prediction in (
            detailed_predictions
        ):

            f.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"\nDetailed predictions saved to:"
    )

    print(output_file)

    return {
        "system": system_name,
        "accuracy": accuracy,
        "macro_precision":
            macro_precision,
        "macro_recall":
            macro_recall,
        "macro_f1":
            macro_f1,
        "weighted_precision":
            weighted_precision,
        "weighted_recall":
            weighted_recall,
        "weighted_f1":
            weighted_f1,
        "confusion_matrix":
            cm.tolist(),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 80)
    print("CONTROLLED RETRIEVAL + NLI EXPERIMENT")
    print("=" * 80)

    print(
        f"Evaluation limit: {EVAL_LIMIT}"
    )

    print(
        f"Retrieval top-K: {RETRIEVAL_TOP_K}"
    )

    print(
        f"NLI top-K: {NLI_TOP_K}"
    )

    print(
        f"NLI batch size: {NLI_BATCH_SIZE}"
    )

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    records = load_jsonl(
        DEV_FILE
    )

    records = records[
        :EVAL_LIMIT
    ]

    print(
        f"\nLoaded {len(records)} evaluation examples."
    )

    # --------------------------------------------------------------
    # Load retrieval models
    # --------------------------------------------------------------

    print("\nLoading retrieval models...")

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = build_models()

    # --------------------------------------------------------------
    # Load NLI model
    # --------------------------------------------------------------

    print("\nLoading NLI model...")

    nli_verifier = NLIVerifier(
        batch_size=NLI_BATCH_SIZE
    )

    # ==============================================================
    # RETRIEVAL FUNCTIONS
    # ==============================================================

    def bm25_retrieval(claim):

        return retrieve_bm25(
            claim,
            bm25,
            bm25_metadata,
            RETRIEVAL_TOP_K
        )

    def semantic_retrieval(claim):

        return retrieve_semantic(
            claim,
            embedding_model,
            semantic_index,
            semantic_metadata,
            RETRIEVAL_TOP_K
        )

    def hybrid_retrieval(claim):

        return hybrid_search(
            claim,
            semantic_index,
            bm25,
            semantic_metadata,
            bm25_metadata,
            embedding_model
        )[:RETRIEVAL_TOP_K]

    # ==============================================================
    # EXPERIMENT 1
    # BM25 + NLI + V2
    # ==============================================================

    bm25_v2 = evaluate_system(
        "BM25 + NLI + V2",
        records,
        bm25_retrieval,
        nli_verifier,
        aggregate_v2,
        RESULTS_DIR
        / "bm25_nli_v2_predictions.jsonl"
    )

    # ==============================================================
    # EXPERIMENT 2
    # SEMANTIC + NLI + V2
    # ==============================================================

    semantic_v2 = evaluate_system(
        "SEMANTIC + NLI + V2",
        records,
        semantic_retrieval,
        nli_verifier,
        aggregate_v2,
        RESULTS_DIR
        / "semantic_nli_v2_predictions.jsonl"
    )

    # ==============================================================
    # EXPERIMENT 3
    # HYBRID + NLI + V2
    # ==============================================================

    hybrid_v2 = evaluate_system(
        "HYBRID + NLI + V2",
        records,
        hybrid_retrieval,
        nli_verifier,
        aggregate_v2,
        RESULTS_DIR
        / "hybrid_nli_v2_predictions.jsonl"
    )

    # ==============================================================
    # EXPERIMENT 4
    # HYBRID + NLI + V1
    # ==============================================================
    #
    # This provides a controlled comparison of the aggregation
    # strategy while keeping retrieval fixed.
    # ==============================================================

    hybrid_v1 = evaluate_system(
        "HYBRID + NLI + V1",
        records,
        hybrid_retrieval,
        nli_verifier,
        aggregate_v1,
        RESULTS_DIR
        / "hybrid_nli_v1_predictions.jsonl"
    )

    # ==============================================================
    # SUMMARY
    # ==============================================================

    all_results = [
        bm25_v2,
        semantic_v2,
        hybrid_v2,
        hybrid_v1,
    ]

    summary_file = (
        RESULTS_DIR
        / "controlled_experiment_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    print("\n")
    print("=" * 80)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 80)

    print(
        f"{'System':<25}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Weighted F1':>15}"
    )

    print("-" * 80)

    for result in all_results:

        print(
            f"{result['system']:<25}"
            f"{result['accuracy']:>12.4f}"
            f"{result['macro_f1']:>12.4f}"
            f"{result['weighted_f1']:>15.4f}"
        )

    print("-" * 80)

    print(
        "\nSummary saved to:"
    )

    print(summary_file)

    print("\nExperiment completed.")


if __name__ == "__main__":

    main()