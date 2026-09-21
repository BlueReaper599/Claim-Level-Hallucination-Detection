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
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE = RAW_DATA_DIR / "shared_task_dev.jsonl"

# Same development subset used in the previous ablation.
EVAL_LIMIT = 1000

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
    Convert the NLI model output into the project's
    three-way classification:

        entailment   -> SUPPORTED
        contradiction -> CONTRADICTED
        neutral       -> UNKNOWN

    A confidence threshold and margin are used to avoid
    making an overly confident decision when the NLI
    probabilities are ambiguous.
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

    # Neutral directly maps to UNKNOWN.
    if best_label == "UNKNOWN":
        final_label = "UNKNOWN"

    # Require both sufficient confidence and separation
    # between the best and second-best class.
    elif (
        best_score >= DECISION_THRESHOLD
        and margin >= DECISION_MARGIN
    ):
        final_label = best_label

    else:
        final_label = "UNKNOWN"

    return final_label, best_score, margin


# ============================================================
# RETRIEVAL METHODS
# ============================================================

def retrieve_bm25(
    claim,
    bm25,
    bm25_metadata,
    top_k=10,
):
    """
    BM25 retrieval.

    IMPORTANT:
    The tokenizer matches the tokenizer used when the
    BM25 index was constructed.
    """

    import re

    tokens = re.findall(
        r"\b\w+\b",
        claim.lower(),
    )

    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        item = bm25_metadata[int(index)]

        results.append({
            "page_id": item["page_id"],
            "sentence_id": item["sentence_id"],
            "text": item["text"],
            "score": float(scores[index]),
        })

    return results


def retrieve_semantic(
    claim,
    semantic_index,
    semantic_metadata,
    embedding_model,
    top_k=10,
):
    """
    Semantic retrieval using MiniLM + FAISS.
    """

    query_embedding = embedding_model.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = semantic_index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0],
    ):
        item = semantic_metadata[int(index)]

        results.append({
            "page_id": item["page_id"],
            "sentence_id": item["sentence_id"],
            "text": item["text"],
            "score": float(score),
        })

    return results


# ============================================================
# EVALUATION
# ============================================================

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

    labels = [
        "SUPPORTED",
        "CONTRADICTED",
        "UNKNOWN",
    ]

    report = classification_report(
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

    print("\n" + "=" * 80)
    print(system_name)
    print("=" * 80)

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall   : {recall_macro:.4f}")
    print(f"Macro F1       : {f1_macro:.4f}")
    print(f"Weighted F1    : {f1_weighted:.4f}")

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(matrix)

    return {
        "system": system_name,
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": matrix.tolist(),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 80)
    print("RETRIEVER ABLATION WITH TOP-1 NLI")
    print("=" * 80)

    print(f"Evaluation file : {EVAL_FILE}")
    print(f"Evaluation limit: {EVAL_LIMIT}")

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    records = load_jsonl(EVAL_FILE)
    records = records[:EVAL_LIMIT]

    print(f"Loaded records: {len(records)}")

    # --------------------------------------------------------
    # Load retrieval models
    # --------------------------------------------------------

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = build_models()

    # --------------------------------------------------------
    # Load NLI model
    # --------------------------------------------------------

    nli_verifier = NLIVerifier(
        batch_size=NLI_BATCH_SIZE
    )

    # --------------------------------------------------------
    # Storage for predictions
    # --------------------------------------------------------

    y_true = []

    predictions_bm25 = []
    predictions_semantic = []
    predictions_hybrid = []

    detailed_results = []

    # --------------------------------------------------------
    # Evaluate every claim
    # --------------------------------------------------------

    for record in tqdm(
        records,
        desc="Evaluating",
    ):

        claim = record["claim"]

        gold_label = LABEL_MAP[
            record["label"]
        ]

        # ====================================================
        # 1. BM25
        # ====================================================

        bm25_results = retrieve_bm25(
            claim,
            bm25,
            bm25_metadata,
            top_k=10,
        )

        bm25_top1 = bm25_results[0]

        bm25_nli = nli_verifier.predict(
            claim,
            bm25_top1["text"],
        )

        bm25_label, bm25_confidence, bm25_margin = (
            classify_nli_result(bm25_nli)
        )

        # ====================================================
        # 2. SEMANTIC
        # ====================================================

        semantic_results = retrieve_semantic(
            claim,
            semantic_index,
            semantic_metadata,
            embedding_model,
            top_k=10,
        )

        semantic_top1 = semantic_results[0]

        semantic_nli = nli_verifier.predict(
            claim,
            semantic_top1["text"],
        )

        semantic_label, semantic_confidence, semantic_margin = (
            classify_nli_result(semantic_nli)
        )

        # ====================================================
        # 3. HYBRID
        # ====================================================

        hybrid_results = hybrid_search(
            claim,
            semantic_index,
            bm25,
            semantic_metadata,
            bm25_metadata,
            embedding_model,
        )

        hybrid_top1 = hybrid_results[0]

        hybrid_nli = nli_verifier.predict(
            claim,
            hybrid_top1["text"],
        )

        hybrid_label, hybrid_confidence, hybrid_margin = (
            classify_nli_result(hybrid_nli)
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        y_true.append(gold_label)

        predictions_bm25.append(
            bm25_label
        )

        predictions_semantic.append(
            semantic_label
        )

        predictions_hybrid.append(
            hybrid_label
        )

        # ----------------------------------------------------
        # Detailed record
        # ----------------------------------------------------

        detailed_results.append({
            "id": record["id"],
            "claim": claim,
            "gold_label": gold_label,

            "bm25": {
                "label": bm25_label,
                "confidence": bm25_confidence,
                "margin": bm25_margin,
                "evidence": {
                    "page_id": bm25_top1["page_id"],
                    "sentence_id": bm25_top1["sentence_id"],
                    "text": bm25_top1["text"],
                    "retrieval_score": bm25_top1["score"],
                },
                "nli": bm25_nli,
            },

            "semantic": {
                "label": semantic_label,
                "confidence": semantic_confidence,
                "margin": semantic_margin,
                "evidence": {
                    "page_id": semantic_top1["page_id"],
                    "sentence_id": semantic_top1["sentence_id"],
                    "text": semantic_top1["text"],
                    "retrieval_score": semantic_top1["score"],
                },
                "nli": semantic_nli,
            },

            "hybrid": {
                "label": hybrid_label,
                "confidence": hybrid_confidence,
                "margin": hybrid_margin,
                "evidence": {
                    "page_id": hybrid_top1["page_id"],
                    "sentence_id": hybrid_top1["sentence_id"],
                    "text": hybrid_top1["text"],
                    "retrieval_score": hybrid_top1["score"],
                },
                "nli": hybrid_nli,
            },
        })

    # ========================================================
    # Evaluate all three systems
    # ========================================================

    results = []

    results.append(
        evaluate_system(
            "BM25 + TOP-1 NLI",
            y_true,
            predictions_bm25,
        )
    )

    results.append(
        evaluate_system(
            "SEMANTIC + TOP-1 NLI",
            y_true,
            predictions_semantic,
        )
    )

    results.append(
        evaluate_system(
            "HYBRID + TOP-1 NLI",
            y_true,
            predictions_hybrid,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"{'System':<28}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Weighted F1':>14}"
    )

    print("-" * 66)

    for result in results:
        print(
            f"{result['system']:<28}"
            f"{result['accuracy']:>12.4f}"
            f"{result['macro_f1']:>12.4f}"
            f"{result['weighted_f1']:>14.4f}"
        )

    # ========================================================
    # Save results
    # ========================================================

    summary_file = (
        RESULTS_DIR /
        "retriever_top1_summary.json"
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
        RESULTS_DIR /
        "retriever_top1_detailed.jsonl"
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
    print(summary_file)
    print(detailed_file)


if __name__ == "__main__":
    main()