from hybrid_retriever import (
    build_models,
    hybrid_search,
)

from nli_verifier import NLIVerifier


# ======================================================================
# CONFIGURATION
# ======================================================================

NLI_TOP_K = 5

DECISION_THRESHOLD = 0.50
DECISION_MARGIN = 0.10


# ======================================================================
# V1: MAXIMUM NLI PROBABILITY
# ======================================================================

def aggregate_v1(nli_results):

    max_entailment = max(
        result["nli"]["probabilities"]["entailment"]
        for result in nli_results
    )

    max_contradiction = max(
        result["nli"]["probabilities"]["contradiction"]
        for result in nli_results
    )

    max_neutral = max(
        result["nli"]["probabilities"]["neutral"]
        for result in nli_results
    )

    if max_contradiction > max_entailment:

        label = "CONTRADICTED"
        confidence = max_contradiction

    elif max_entailment > max_contradiction:

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
# V2: RANK-WEIGHTED NLI
# ======================================================================

def aggregate_v2(nli_results):

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
            result["nli"]["probabilities"]
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


# ======================================================================
# V3: RETRIEVAL-WEIGHTED NLI
# ======================================================================

def aggregate_v3(nli_results):

    if not nli_results:

        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "margin": 0.0,
            "scores": {
                "SUPPORTED": 0.0,
                "CONTRADICTED": 0.0,
                "UNKNOWN": 1.0,
            },
        }

    retrieval_scores = [
        float(
            result.get("retrieval_score", 0.0)
        )
        for result in nli_results
    ]

    total_retrieval_score = sum(
        retrieval_scores
    )

    # Safety fallback.
    if total_retrieval_score <= 0:

        weights = [
            1.0 / len(nli_results)
            for _ in nli_results
        ]

    else:

        weights = [
            score / total_retrieval_score
            for score in retrieval_scores
        ]

    weighted_entailment = 0.0
    weighted_contradiction = 0.0
    weighted_neutral = 0.0

    for weight, result in zip(
        weights,
        nli_results
    ):

        probabilities = (
            result["nli"]["probabilities"]
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

    scores = {
        "SUPPORTED": weighted_entailment,
        "CONTRADICTED": weighted_contradiction,
        "UNKNOWN": weighted_neutral,
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


# ======================================================================
# CLAIM VERIFICATION
# ======================================================================

def verify_claim(
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

    retrieved = retrieved[
        :NLI_TOP_K
    ]

    evidence_texts = [
        item["text"]
        for item in retrieved
    ]

    nli_results = (
        nli_verifier.predict_batch(
            claim,
            evidence_texts,
        )
    )

    combined_results = []

    for item, nli_result in zip(
        retrieved,
        nli_results,
    ):

        combined_results.append(
            {
                "page_id": item["page_id"],
                "sentence_id": item["sentence_id"],
                "text": item["text"],

                # Actual field returned by hybrid_search.
                "retrieval_score": item["score"],

                "nli": nli_result,
            }
        )

    v1 = aggregate_v1(
        combined_results
    )

    v2 = aggregate_v2(
        combined_results
    )

    v3 = aggregate_v3(
        combined_results
    )

    return {
        "claim": claim,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "evidence": combined_results,
    }


# ======================================================================
# PRINT RESULT
# ======================================================================

def print_result(result):

    print("\n" + "=" * 80)

    print("CLAIM:")
    print(result["claim"])

    print("\n" + "-" * 80)

    print("V1:")
    print(result["v1"])

    print("\nV2:")
    print(result["v2"])

    print("\nV3:")
    print(result["v3"])

    print("\n" + "-" * 80)

    print("EVIDENCE:")

    for i, evidence in enumerate(
        result["evidence"],
        start=1,
    ):

        print(
            f"\nEvidence {i}"
        )

        print(
            f"Page: "
            f"{evidence['page_id']}"
        )

        print(
            f"Retrieval score: "
            f"{evidence['retrieval_score']}"
        )

        print(
            f"NLI: "
            f"{evidence['nli']['label']}"
        )

        print(
            "Probabilities: "
            f"{evidence['nli']['probabilities']}"
        )

        print(
            f"Text: "
            f"{evidence['text']}"
        )


# ======================================================================
# MAIN
# ======================================================================

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

    test_claims = [

        "Paris is the capital of France.",

        "Paris is the capital of Germany.",

        "Paris has exactly 10 million inhabitants.",

    ]

    for claim in test_claims:

        result = verify_claim(
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