from nli_verifier import NLIVerifier


def main():

    verifier = NLIVerifier(
        batch_size=16
    )

    claim = (
        "Paris is the capital of France."
    )

    evidence_list = [

        "Paris is the capital and "
        "largest city of France.",

        "Berlin is the capital of Germany.",

        "Paris is located in France.",

        "The Eiffel Tower is in Paris.",
    ]

    print("=" * 70)
    print("SINGLE VS BATCH NLI TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Single predictions
    # ---------------------------------------------------------------

    single_results = []

    for evidence in evidence_list:

        result = verifier.predict(
            claim,
            evidence
        )

        single_results.append(
            result
        )

    # ---------------------------------------------------------------
    # Batch prediction
    # ---------------------------------------------------------------

    batch_results = (
        verifier.predict_batch(
            claim,
            evidence_list
        )
    )

    # ---------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------

    print("\nComparison:")

    all_match = True

    for i, (
        single,
        batch
    ) in enumerate(
        zip(
            single_results,
            batch_results
        ),
        start=1
    ):

        label_match = (
            single["label"]
            == batch["label"]
        )

        single_probs = (
            single["probabilities"]
        )

        batch_probs = (
            batch["probabilities"]
        )

        probability_match = all(
            abs(
                single_probs[label]
                - batch_probs[label]
            ) < 1e-5
            for label in single_probs
        )

        match = (
            label_match
            and probability_match
        )

        if not match:

            all_match = False

        print(
            f"Evidence {i}: "
            f"{'MATCH' if match else 'MISMATCH'}"
        )

        print(
            f"  Single: {single}"
        )

        print(
            f"  Batch : {batch}"
        )

    print("\n" + "=" * 70)

    if all_match:

        print(
            "RESULT: Single and batch predictions match."
        )

    else:

        print(
            "RESULT: There are differences."
        )

    print("=" * 70)


if __name__ == "__main__":

    main()