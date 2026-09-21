import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "halueval"
    / "qa_data.json"
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Development-scale external evaluation first.
MAX_EXAMPLES = 1000

BATCH_SIZE = 16

ENTAILMENT_THRESHOLD = 0.50
DECISION_MARGIN = 0.10


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_halueval():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"HaluEval file not found:\n{INPUT_FILE}"
        )

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


# ---------------------------------------------------------------------
# NLI verifier
# ---------------------------------------------------------------------

class NLIVerifier:

    def __init__(self):

        print("=" * 70)
        print("Loading HaluEval NLI verifier")
        print("=" * 70)

        print(f"Model : {MODEL_NAME}")
        print(f"Device: {DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        self.model.to(DEVICE)
        self.model.eval()

        print("\nModel labels:")
        print(self.model.config.id2label)

        print("\nNLI verifier ready.")

    def predict_batch(self, premises, hypotheses):

        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        return probabilities.detach().cpu().numpy()


# ---------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------

def classify_answer(probabilities):

    contradiction = float(probabilities[0])
    entailment = float(probabilities[1])
    neutral = float(probabilities[2])

    competing_score = max(
        contradiction,
        neutral,
    )

    margin = entailment - competing_score

    if (
        entailment >= ENTAILMENT_THRESHOLD
        and margin >= DECISION_MARGIN
    ):
        label = "SUPPORTED"
    else:
        label = "UNVERIFIED"

    return {
        "label": label,
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": neutral,
        "margin": margin,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate():

    print("=" * 80)
    print("HALUEVAL EXTERNAL BENCHMARK")
    print("QUESTION-AWARE NLI EVALUATION")
    print("=" * 80)

    records = load_halueval()

    total_available = len(records)

    print(
        f"\nTotal HaluEval examples available: "
        f"{total_available}"
    )

    if MAX_EXAMPLES is not None:
        records = records[:MAX_EXAMPLES]

    print(
        f"Examples evaluated: {len(records)}"
    )

    verifier = NLIVerifier()

    premises = []
    hypotheses = []
    labels = []
    example_ids = []

    # -------------------------------------------------------------
    # Construct question-aware NLI pairs
    # -------------------------------------------------------------

    for index, record in enumerate(records):

        knowledge = record["knowledge"]
        question = record["question"]

        right_answer = record["right_answer"]
        hallucinated_answer = record["hallucinated_answer"]

        premise = (
            "Knowledge:\n"
            + knowledge
            + "\n\n"
            + "Question:\n"
            + question
        )

        # Grounded answer
        premises.append(premise)
        hypotheses.append(right_answer)
        labels.append(1)
        example_ids.append(
            f"{index}_right"
        )

        # Hallucinated answer
        premises.append(premise)
        hypotheses.append(hallucinated_answer)
        labels.append(0)
        example_ids.append(
            f"{index}_hallucinated"
        )

    # -------------------------------------------------------------
    # NLI inference
    # -------------------------------------------------------------

    all_probabilities = []

    print("\nRunning question-aware NLI inference...")

    for start in tqdm(
        range(
            0,
            len(premises),
            BATCH_SIZE,
        )
    ):

        batch_premises = premises[
            start:start + BATCH_SIZE
        ]

        batch_hypotheses = hypotheses[
            start:start + BATCH_SIZE
        ]

        probabilities = verifier.predict_batch(
            batch_premises,
            batch_hypotheses,
        )

        all_probabilities.append(
            probabilities
        )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    # -------------------------------------------------------------
    # Decisions
    # -------------------------------------------------------------

    predictions = []
    entailment_scores = []
    detailed_results = []

    for index in range(len(labels)):

        decision = classify_answer(
            probabilities[index]
        )

        prediction = (
            1
            if decision["label"] == "SUPPORTED"
            else 0
        )

        predictions.append(prediction)

        entailment_scores.append(
            decision["entailment"]
        )

        detailed_results.append(
            {
                "example_id": example_ids[index],
                "gold_label": (
                    "SUPPORTED"
                    if labels[index] == 1
                    else "HALLUCINATED"
                ),
                "predicted_label": decision["label"],
                "entailment": decision["entailment"],
                "contradiction": decision["contradiction"],
                "neutral": decision["neutral"],
                "margin": decision["margin"],
            }
        )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    roc_auc = roc_auc_score(
        labels,
        entailment_scores,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    report = classification_report(
        labels,
        predictions,
        target_names=[
            "HALLUCINATED",
            "SUPPORTED",
        ],
        zero_division=0,
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("QUESTION-AWARE HALUEVAL RESULTS")
    print("=" * 80)

    print(
        f"\nExamples: {len(records)}"
    )

    print(
        f"Answer pairs evaluated: {len(labels)}"
    )

    print(
        f"\nAccuracy:           {accuracy:.4f}"
    )

    print(
        f"Precision:          {precision:.4f}"
    )

    print(
        f"Recall:             {recall:.4f}"
    )

    print(
        f"F1:                 {f1:.4f}"
    )

    print(
        f"ROC-AUC:            {roc_auc:.4f}"
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(report)

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "dataset": "HaluEval QA",
        "evaluation": "question-aware NLI",
        "total_available_examples": total_available,
        "examples_evaluated": len(records),
        "answer_pairs_evaluated": len(labels),
        "model": MODEL_NAME,
        "device": DEVICE,
        "entailment_threshold": ENTAILMENT_THRESHOLD,
        "decision_margin": DECISION_MARGIN,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }

    summary_file = (
        RESULTS_DIR
        / "halueval_question_aware_summary.json"
    )

    detailed_file = (
        RESULTS_DIR
        / "halueval_question_aware_detailed.jsonl"
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

    print(
        f"\nSummary saved to:\n{summary_file}"
    )

    print(
        f"Detailed results saved to:\n{detailed_file}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()