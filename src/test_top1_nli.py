from hybrid_retriever import (
    build_models,
    hybrid_search,
)

from nli_verifier import NLIVerifier


# ============================================================
# CONFIGURATION
# ============================================================

DECISION_THRESHOLD = 0.50
DECISION_MARGIN = 0.10


# ============================================================
# TOP-1 NLI DECISION
# ============================================================

def classify_top1(nli_result):

    probabilities = (
        nli_result["probabilities"]
    )

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

    margin = (
        best_score
        - second_score
    )

    if best_label == "UNKNOWN":

        final_label = "UNKNOWN"

    elif (
        best_score >= DECISION_THRESHOLD
        and margin >= DECISION_MARGIN
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


# ============================================================
# VERIFY ONE CLAIM
# ============================================================

def verify_top1(
    claim,
    semantic_index,
    bm25,
    semantic_metadata,
    bm25_metadata,
    embedding_model,
    nli_verifier,
):

    retrieved = hybrid_search(
        claim,
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    )

    top_evidence = retrieved[0]

    nli_result = nli_verifier.predict(
        claim,
        top_evidence["text"],
    )

    decision = classify_top1(
        nli_result
    )

    return {
        "claim": claim,
        "decision": decision,
        "evidence": top_evidence,
        "nli": nli_result,
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print("\n" + "=" * 80)

    print("CLAIM")
    print("=" * 80)

    print(
        result["claim"]
    )

    print("\n" + "-" * 80)

    print("TOP RETRIEVED EVIDENCE")
    print("-" * 80)

    evidence = result["evidence"]

    print(
        f"Page: {evidence['page_id']}"
    )

    print(
        f"Sentence ID: "
        f"{evidence['sentence_id']}"
    )

    print(
        f"Retrieval score: "
        f"{evidence['score']}"
    )

    print(
        f"Text: {evidence['text']}"
    )

    print("\n" + "-" * 80)

    print("NLI")
    print("-" * 80)

    print(
        f"Label: "
        f"{result['nli']['label']}"
    )

    print(
        f"Probabilities: "
        f"{result['nli']['probabilities']}"
    )

    print("\n" + "-" * 80)

    print("FINAL TOP-1 DECISION")
    print("-" * 80)

    print(
        result["decision"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = build_models()

    nli_verifier = NLIVerifier(
        batch_size=16
    )

    claims = [

        "Paris is the capital of France.",

        "Paris is the capital of Germany.",

        "Paris has exactly 10 million inhabitants.",

    ]

    for claim in claims:

        result = verify_top1(
            claim,
            semantic_index,
            bm25,
            semantic_metadata,
            bm25_metadata,
            embedding_model,
            nli_verifier,
        )

        print_result(
            result
        )


if __name__ == "__main__":

    main()