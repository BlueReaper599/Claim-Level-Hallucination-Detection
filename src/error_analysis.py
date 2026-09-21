import json
from pathlib import Path

from hybrid_retriever import build_models, hybrid_search
from nli_verifier import NLIVerifier


# =====================================================================
# PATHS
# =====================================================================

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

OUTPUT_FILE = (
    RESULTS_DIR
    / "error_analysis.jsonl"
)


# =====================================================================
# PARAMETERS
# =====================================================================

EVAL_LIMIT = 100

RETRIEVAL_TOP_K = 10

NLI_TOP_K = 5

DECISION_THRESHOLD = 0.50

DECISION_MARGIN = 0.10


# =====================================================================
# LABEL MAPPING
# =====================================================================

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "UNKNOWN",
}


# =====================================================================
# DATA LOADING
# =====================================================================

def load_jsonl(filename):

    records = []

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    return records


# =====================================================================
# GOLD EVIDENCE
# =====================================================================

def get_gold_sentences(record):

    gold = set()

    for evidence_group in record.get(
        "evidence",
        []
    ):

        for evidence_item in evidence_group:

            if len(evidence_item) >= 4:

                page_id = evidence_item[2]

                sentence_id = evidence_item[3]

                if (
                    page_id is not None
                    and sentence_id is not None
                ):

                    gold.add(
                        (
                            str(page_id),
                            str(sentence_id)
                        )
                    )

    return gold


# =====================================================================
# V2 AGGREGATION
# =====================================================================

def aggregate_predictions(
    nli_results
):

    weighted_entailment = 0.0
    weighted_contradiction = 0.0
    weighted_neutral = 0.0

    for rank, result in enumerate(
        nli_results,
        start=1
    ):

        weight = 1.0 / rank

        probabilities = (
            result["nli"]["probabilities"]
        )

        weighted_entailment += (
            probabilities["entailment"]
            * weight
        )

        weighted_contradiction += (
            probabilities["contradiction"]
            * weight
        )

        weighted_neutral += (
            probabilities["neutral"]
            * weight
        )

    total = (
        weighted_entailment
        + weighted_contradiction
        + weighted_neutral
    )

    if total == 0:

        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "margin": 0.0,
            "scores": {
                "SUPPORTED": 0.0,
                "CONTRADICTED": 0.0,
                "UNKNOWN": 0.0,
            },
        }

    scores = {

        "SUPPORTED":
            weighted_entailment / total,

        "CONTRADICTED":
            weighted_contradiction / total,

        "UNKNOWN":
            weighted_neutral / total,
    }

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_label = ranked[0][0]

    best_score = ranked[0][1]

    second_score = ranked[1][1]

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


# =====================================================================
# NLI
# =====================================================================

def run_nli(
    verifier,
    claim,
    evidence_results
):

    nli_results = []

    for evidence in evidence_results[
        :NLI_TOP_K
    ]:

        nli = verifier.predict(
            claim,
            evidence["text"]
        )

        nli_results.append({

            "evidence": evidence,

            "nli": nli,
        })

    return nli_results


# =====================================================================
# GOLD RETRIEVAL CHECK
# =====================================================================

def check_gold_retrieval(
    retrieved_results,
    gold_sentences
):

    if not gold_sentences:

        return None

    retrieved = set()

    for result in retrieved_results:

        retrieved.add(
            (
                result["page_id"],
                str(result["sentence_id"])
            )
        )

    return bool(
        retrieved
        & gold_sentences
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 80)
    print("ERROR ATTRIBUTION EXPERIMENT")
    print("=" * 80)

    print(
        f"\nEvaluation size: {EVAL_LIMIT}"
    )

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    data = load_jsonl(
        DEV_FILE
    )

    data = data[:EVAL_LIMIT]

    # ---------------------------------------------------------------
    # Load retrieval models
    # ---------------------------------------------------------------

    print("\nLoading retrieval models...")

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = build_models()

    # ---------------------------------------------------------------
    # Load NLI
    # ---------------------------------------------------------------

    print("\nLoading NLI verifier...")

    verifier = NLIVerifier()

    # ---------------------------------------------------------------
    # Counters
    # ---------------------------------------------------------------

    category_counts = {

        "CORRECT": 0,

        "RETRIEVAL_MISS": 0,

        "NLI_OR_VERIFICATION_ERROR": 0,

        "UNKNOWN_ERROR_TYPE": 0,
    }

    results = []

    # ---------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------

    for i, record in enumerate(
        data,
        start=1
    ):

        claim = record["claim"]

        gold_label = LABEL_MAP[
            record["label"]
        ]

        gold_sentences = (
            get_gold_sentences(
                record
            )
        )

        # -----------------------------------------------------------
        # Hybrid retrieval
        #
        # IMPORTANT:
        # This matches the existing hybrid_search()
        # implementation in your project.
        # -----------------------------------------------------------

        retrieved = hybrid_search(
            claim,
            semantic_index,
            bm25,
            semantic_metadata,
            bm25_metadata,
            embedding_model
        )

        # Restrict to the top K returned by the retriever.

        retrieved = retrieved[
            :RETRIEVAL_TOP_K
        ]

        # -----------------------------------------------------------
        # Was gold evidence retrieved?
        # -----------------------------------------------------------

        gold_retrieved = (
            check_gold_retrieval(
                retrieved,
                gold_sentences
            )
        )

        # -----------------------------------------------------------
        # NLI
        # -----------------------------------------------------------

        nli_results = run_nli(
            verifier,
            claim,
            retrieved
        )

        # -----------------------------------------------------------
        # V2 decision
        # -----------------------------------------------------------

        decision = aggregate_predictions(
            nli_results
        )

        predicted_label = decision[
            "label"
        ]

        # -----------------------------------------------------------
        # Correct?
        # -----------------------------------------------------------

        correct = (
            predicted_label
            == gold_label
        )

        # -----------------------------------------------------------
        # Error attribution
        # -----------------------------------------------------------

        if correct:

            category = "CORRECT"

        elif (
            gold_retrieved is False
            and gold_sentences
        ):

            category = "RETRIEVAL_MISS"

        elif gold_retrieved is True:

            category = (
                "NLI_OR_VERIFICATION_ERROR"
            )

        else:

            # This primarily represents NEI
            # examples for which FEVER does not
            # provide sentence-level gold evidence.

            category = (
                "UNKNOWN_ERROR_TYPE"
            )

        category_counts[
            category
        ] += 1

        # -----------------------------------------------------------
        # Store detailed result
        # -----------------------------------------------------------

        output_record = {

            "id":
                record["id"],

            "claim":
                claim,

            "gold_label":
                gold_label,

            "predicted_label":
                predicted_label,

            "correct":
                correct,

            "error_category":
                category,

            "gold_sentence_count":
                len(gold_sentences),

            "gold_evidence_retrieved":
                gold_retrieved,

            "decision":
                decision,

            "retrieved_evidence":
                retrieved,

            "nli_results":
                nli_results,
        }

        results.append(
            output_record
        )

        print(
            f"[{i:>3}/{len(data)}] "
            f"{gold_label:<13} -> "
            f"{predicted_label:<13} "
            f"{category}"
        )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for record in results:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ERROR ATTRIBUTION SUMMARY")
    print("=" * 80)

    total = len(results)

    for category, count in (
        category_counts.items()
    ):

        percentage = (
            count / total * 100
        )

        print(
            f"{category:<35}"
            f"{count:>5}"
            f" ({percentage:>6.2f}%)"
        )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        "\nCORRECT:"
        " final V2 classification matched FEVER."
    )

    print(
        "\nRETRIEVAL_MISS:"
        " usable gold evidence existed, but "
        "the hybrid retriever did not retrieve "
        "it in the top-10."
    )

    print(
        "\nNLI_OR_VERIFICATION_ERROR:"
        " gold evidence was retrieved, but "
        "the final prediction was incorrect."
    )

    print(
        "\nUNKNOWN_ERROR_TYPE:"
        " primarily applies to NEI examples "
        "where FEVER does not provide "
        "sentence-level gold evidence."
    )

    print(
        f"\nDetailed results saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()