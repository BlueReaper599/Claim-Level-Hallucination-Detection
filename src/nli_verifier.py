import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class NLIVerifier:

    def __init__(
        self,
        batch_size=16
    ):

        print("=" * 70)
        print("Loading NLI model")
        print("=" * 70)

        print(f"Model : {MODEL_NAME}")
        print(f"Device: {DEVICE}")

        self.batch_size = batch_size

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL_NAME)
        )

        self.model.to(DEVICE)

        self.model.eval()

        print(
            "NLI model loaded successfully."
        )

        print("\nModel labels:")
        print(
            self.model.config.id2label
        )

    # ================================================================
    # SINGLE PREDICTION
    # ================================================================

    def predict(
        self,
        claim,
        evidence
    ):

        results = self.predict_batch(
            claim,
            [evidence]
        )

        return results[0]

    # ================================================================
    # BATCH PREDICTION
    # ================================================================

    def predict_batch(
        self,
        claim,
        evidence_list
    ):

        if not evidence_list:

            return []

        claims = [
            claim
            for _ in evidence_list
        ]

        all_results = []

        for start in range(
            0,
            len(evidence_list),
            self.batch_size
        ):

            batch_evidence = (
                evidence_list[
                    start:
                    start + self.batch_size
                ]
            )

            batch_claims = (
                claims[
                    start:
                    start + self.batch_size
                ]
            )

            inputs = self.tokenizer(
                batch_evidence,
                batch_claims,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            predicted_ids = (
                torch.argmax(
                    probabilities,
                    dim=-1
                )
            )

            for i in range(
                len(batch_evidence)
            ):

                predicted_id = int(
                    predicted_ids[i]
                )

                result = {

                    "label":
                        self.model.config.id2label[
                            predicted_id
                        ],

                    "probabilities": {

                        self.model.config.id2label[
                            j
                        ]:
                            float(
                                probabilities[
                                    i,
                                    j
                                ]
                            )

                        for j in range(
                            probabilities.shape[1]
                        )
                    }
                }

                all_results.append(
                    result
                )

        return all_results


# ====================================================================
# TEST
# ====================================================================

if __name__ == "__main__":

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
    ]

    print("\n" + "=" * 70)
    print("BATCH NLI TEST")
    print("=" * 70)

    results = verifier.predict_batch(
        claim,
        evidence_list
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nEvidence {i}:"
        )

        print(
            result
        )