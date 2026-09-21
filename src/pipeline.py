import re
from pathlib import Path

from hybrid_retriever import build_models, hybrid_search
from nli_verifier import NLIVerifier


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DECISION_THRESHOLD = 0.50
DECISION_MARGIN = 0.10


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claims(text):
    """
    Lightweight deterministic claim extraction.

    For the current version, each sufficiently long sentence
    is treated as an individual factual claim.

    This deliberately avoids introducing another large
    language model into the pipeline.

    Returns:
        list[str]
    """

    if not text or not text.strip():
        return []

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text.strip())

    # Basic sentence splitting.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    claims = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Remove accidental leading/trailing punctuation.
        sentence = sentence.strip(" \t\n")

        # Avoid extremely short fragments.
        if len(sentence.split()) < 3:
            continue

        claims.append(sentence)

    return claims


# ============================================================
# NLI DECISION
# ============================================================

def classify_nli_result(nli_result):
    """
    Convert DeBERTa NLI output into:

        SUPPORTED
        CONTRADICTED
        UNKNOWN
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

    return {
        "label": final_label,
        "confidence": float(best_score),
        "margin": float(margin),
        "probabilities": {
            key: float(value)
            for key, value in probabilities.items()
        },
    }


# ============================================================
# VERIFIER
# ============================================================

class HallucinationDetectionPipeline:

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING HALLUCINATION DETECTION PIPELINE")
        print("=" * 70)

        print("\nLoading retrieval system...")

        (
            self.semantic_index,
            self.bm25,
            self.semantic_metadata,
            self.bm25_metadata,
            self.embedding_model,
        ) = build_models()

        print("\nLoading NLI verifier...")

        self.nli_verifier = NLIVerifier(
            batch_size=16
        )

        print("\nPipeline ready.")

    # --------------------------------------------------------
    # Verify one claim
    # --------------------------------------------------------

    def verify_claim(self, claim):

        if not claim or not claim.strip():
            raise ValueError(
                "Claim cannot be empty."
            )

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        retrieved = hybrid_search(
            claim,
            self.semantic_index,
            self.bm25,
            self.semantic_metadata,
            self.bm25_metadata,
            self.embedding_model,
        )

        if not retrieved:
            return {
                "claim": claim,
                "label": "UNKNOWN",
                "confidence": 0.0,
                "margin": 0.0,
                "probabilities": {},
                "evidence": None,
            }

        # ----------------------------------------------------
        # Frozen architecture:
        #
        # Hybrid retrieval
        #       ↓
        # Top-1 evidence
        #       ↓
        # NLI
        # ----------------------------------------------------

        top_evidence = retrieved[0]

        nli_result = self.nli_verifier.predict(
            claim,
            top_evidence["text"],
        )

        decision = classify_nli_result(
            nli_result
        )

        return {
            "claim": claim,

            "label": decision["label"],

            "confidence": decision[
                "confidence"
            ],

            "margin": decision[
                "margin"
            ],

            "probabilities": decision[
                "probabilities"
            ],

            "evidence": {
                "page_id": top_evidence[
                    "page_id"
                ],

                "sentence_id": top_evidence[
                    "sentence_id"
                ],

                "text": top_evidence[
                    "text"
                ],

                "retrieval_score": float(
                    top_evidence["score"]
                ),
            },
        }

    # --------------------------------------------------------
    # Verify complete answer
    # --------------------------------------------------------

    def verify_answer(self, answer):

        if not answer or not answer.strip():
            raise ValueError(
                "Answer cannot be empty."
            )

        claims = extract_claims(answer)

        results = []

        for claim in claims:

            result = self.verify_claim(
                claim
            )

            results.append(result)

        # ----------------------------------------------------
        # Overall summary
        # ----------------------------------------------------

        counts = {
            "SUPPORTED": 0,
            "CONTRADICTED": 0,
            "UNKNOWN": 0,
        }

        for result in results:
            counts[result["label"]] += 1

        total = len(results)

        if total > 0:
            hallucination_rate = (
                counts["CONTRADICTED"]
                / total
            )
        else:
            hallucination_rate = 0.0

        return {
            "answer": answer,

            "claims": results,

            "summary": {
                "total_claims": total,

                "supported": counts[
                    "SUPPORTED"
                ],

                "contradicted": counts[
                    "CONTRADICTED"
                ],

                "unknown": counts[
                    "UNKNOWN"
                ],

                "contradicted_fraction": (
                    float(hallucination_rate)
                ),
            },
        }


# ============================================================
# SIMPLE COMMAND-LINE DEMO
# ============================================================

def print_result(result):

    print("\n" + "=" * 80)
    print("VERIFICATION RESULT")
    print("=" * 80)

    print(
        f"Claim: {result['claim']}"
    )

    print(
        f"\nDecision: {result['label']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Margin: "
        f"{result['margin']:.4f}"
    )

    print("\nNLI probabilities:")

    for label, probability in (
        result["probabilities"].items()
    ):

        print(
            f"  {label:<15}"
            f"{probability:.4f}"
        )

    print("\nEvidence:")

    if result["evidence"]:

        evidence = result["evidence"]

        print(
            f"Page: "
            f"{evidence['page_id']}"
        )

        print(
            f"Sentence: "
            f"{evidence['sentence_id']}"
        )

        print(
            f"Retrieval score: "
            f"{evidence['retrieval_score']:.6f}"
        )

        print(
            f"Text: "
            f"{evidence['text']}"
        )

    else:

        print("No evidence retrieved.")


def main():

    pipeline = (
        HallucinationDetectionPipeline()
    )

    # --------------------------------------------------------
    # Single-claim demonstration
    # --------------------------------------------------------

    test_claims = [
        "Paris is the capital of France.",
        "Paris is the capital of Germany.",
        "Paris has exactly 10 million inhabitants.",
    ]

    for claim in test_claims:

        result = pipeline.verify_claim(
            claim
        )

        print_result(result)

    # --------------------------------------------------------
    # Full-answer demonstration
    # --------------------------------------------------------

    answer = (
        "Paris is the capital of France. "
        "It is located in Germany. "
        "The city has a long history."
    )

    print("\n" + "=" * 80)
    print("FULL ANSWER TEST")
    print("=" * 80)

    answer_result = pipeline.verify_answer(
        answer
    )

    print(
        f"\nTotal claims: "
        f"{answer_result['summary']['total_claims']}"
    )

    print(
        f"Supported: "
        f"{answer_result['summary']['supported']}"
    )

    print(
        f"Contradicted: "
        f"{answer_result['summary']['contradicted']}"
    )

    print(
        f"Unknown: "
        f"{answer_result['summary']['unknown']}"
    )

    for claim_result in (
        answer_result["claims"]
    ):

        print_result(
            claim_result
        )


if __name__ == "__main__":
    main()