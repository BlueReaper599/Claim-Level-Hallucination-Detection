import sys
from pathlib import Path

import gradio as gr


# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import HallucinationDetectionPipeline


# ---------------------------------------------------------------------
# Load the ML pipeline ONCE
# ---------------------------------------------------------------------

print("=" * 80)
print("STARTING TRUTHLENS")
print("=" * 80)

pipeline = HallucinationDetectionPipeline()


# ---------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------

def verify_answer(answer):
    if not answer or not answer.strip():
        return (
            "⚠️ Please enter an AI-generated answer to verify.",
            "",
        )

    try:
        result = pipeline.verify_answer(answer)

        summary = result["summary"]
        claims = result["claims"]

        # -------------------------------------------------------------
        # Summary
        # -------------------------------------------------------------

        summary_html = f"""
        <div class="summary-card">
            <h2>Verification Summary</h2>

            <div class="summary-grid">

                <div class="summary-item">
                    <div class="summary-number">
                        {summary["total_claims"]}
                    </div>
                    <div class="summary-label">
                        Claims
                    </div>
                </div>

                <div class="summary-item supported-card">
                    <div class="summary-number">
                        {summary["supported"]}
                    </div>
                    <div class="summary-label">
                        Supported
                    </div>
                </div>

                <div class="summary-item contradicted-card">
                    <div class="summary-number">
                        {summary["contradicted"]}
                    </div>
                    <div class="summary-label">
                        Contradicted
                    </div>
                </div>

                <div class="summary-item unknown-card">
                    <div class="summary-number">
                        {summary["unknown"]}
                    </div>
                    <div class="summary-label">
                        Unknown
                    </div>
                </div>

            </div>

            <p class="summary-note">
                The contradicted fraction is the proportion of extracted
                claims classified as CONTRADICTED. It is not a universal
                measure of factuality or a guarantee that the answer is
                hallucinated.
            </p>
        </div>
        """

        # -------------------------------------------------------------
        # Individual claim results
        # -------------------------------------------------------------

        result_html = ""

        for index, claim_result in enumerate(claims, start=1):

            label = claim_result["label"]
            confidence = claim_result["confidence"]
            margin = claim_result["margin"]

            if label == "SUPPORTED":
                icon = "✓"
                css_class = "supported"
                label_text = "SUPPORTED"

            elif label == "CONTRADICTED":
                icon = "✗"
                css_class = "contradicted"
                label_text = "CONTRADICTED"

            else:
                icon = "?"
                css_class = "unknown"
                label_text = "UNKNOWN"

            probabilities = claim_result["probabilities"]

            contradiction_probability = probabilities.get(
                "contradiction",
                0.0,
            )

            entailment_probability = probabilities.get(
                "entailment",
                0.0,
            )

            neutral_probability = probabilities.get(
                "neutral",
                0.0,
            )

            evidence = claim_result.get("evidence")

            if evidence:
                evidence_html = f"""
                <div class="evidence-box">

                    <h4>Retrieved Evidence</h4>

                    <p>
                        <strong>Wikipedia page:</strong>
                        {evidence["page_id"]}
                    </p>

                    <p>
                        <strong>Sentence:</strong>
                        {evidence["sentence_id"]}
                    </p>

                    <blockquote>
                        {evidence["text"]}
                    </blockquote>

                    <p class="small-text">
                        Retrieval score:
                        {evidence["retrieval_score"]:.6f}
                    </p>

                </div>
                """
            else:
                evidence_html = """
                <div class="evidence-box">
                    <h4>Retrieved Evidence</h4>
                    <p>No evidence was retrieved.</p>
                </div>
                """

            result_html += f"""
            <div class="claim-card {css_class}">

                <div class="claim-header">

                    <div class="claim-number">
                        Claim {index}
                    </div>

                    <div class="decision {css_class}">
                        <span class="decision-icon">
                            {icon}
                        </span>

                        {label_text}
                    </div>

                </div>

                <div class="claim-text">
                    {claim_result["claim"]}
                </div>

                <div class="metrics">

                    <div>
                        <strong>Confidence</strong>
                        <br>
                        {confidence * 100:.2f}%
                    </div>

                    <div>
                        <strong>Decision margin</strong>
                        <br>
                        {margin * 100:.2f}%
                    </div>

                    <div>
                        <strong>Entailment</strong>
                        <br>
                        {entailment_probability * 100:.2f}%
                    </div>

                    <div>
                        <strong>Contradiction</strong>
                        <br>
                        {contradiction_probability * 100:.2f}%
                    </div>

                    <div>
                        <strong>Neutral</strong>
                        <br>
                        {neutral_probability * 100:.2f}%
                    </div>

                </div>

                {evidence_html}

            </div>
            """

        return summary_html, result_html

    except Exception as error:
        error_html = f"""
        <div class="error-box">
            <h3>Verification error</h3>
            <p>{str(error)}</p>
        </div>
        """

        return error_html, ""


# ---------------------------------------------------------------------
# Example inputs
# ---------------------------------------------------------------------

EXAMPLE_ANSWER = (
    "Paris is the capital of France. "
    "It is located in Germany. "
    "The city has a long history."
)


# ---------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------

CSS = """

body {
    font-family: Arial, sans-serif;
}

#main-container {
    max-width: 1100px;
    margin: auto;
}

.hero {
    text-align: center;
    padding: 20px 10px 10px 10px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.8;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.summary-card {
    padding: 20px;
    border-radius: 14px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.summary-card h2 {
    margin-top: 0;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.summary-item {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid rgba(128, 128, 128, 0.3);
}

.summary-number {
    font-size: 28px;
    font-weight: bold;
}

.summary-label {
    margin-top: 5px;
    font-size: 14px;
}

.supported-card {
    border-left: 5px solid #2e8b57;
}

.contradicted-card {
    border-left: 5px solid #d9534f;
}

.unknown-card {
    border-left: 5px solid #e0a800;
}

.summary-note {
    font-size: 13px;
    opacity: 0.75;
    margin-top: 15px;
}

.claim-card {
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 18px;
    border-left: 7px solid;
}

.claim-card.supported {
    border-left-color: #2e8b57;
}

.claim-card.contradicted {
    border-left-color: #d9534f;
}

.claim-card.unknown {
    border-left-color: #e0a800;
}

.claim-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
}

.claim-number {
    font-weight: bold;
    font-size: 14px;
    opacity: 0.7;
}

.decision {
    font-weight: bold;
    padding: 7px 12px;
    border-radius: 8px;
}

.decision.supported {
    color: #2e8b57;
}

.decision.contradicted {
    color: #d9534f;
}

.decision.unknown {
    color: #b8860b;
}

.decision-icon {
    font-size: 18px;
    margin-right: 5px;
}

.claim-text {
    font-size: 20px;
    margin-top: 15px;
    margin-bottom: 18px;
    font-weight: 500;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}

.metrics > div {
    padding: 10px;
    border-radius: 8px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    text-align: center;
    font-size: 13px;
}

.evidence-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.25);
}

.evidence-box h4 {
    margin-top: 0;
}

.evidence-box blockquote {
    margin: 10px 0;
    padding: 12px;
    border-left: 4px solid rgba(128, 128, 128, 0.5);
    font-style: italic;
}

.small-text {
    font-size: 12px;
    opacity: 0.65;
}

.error-box {
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #d9534f;
}

@media (max-width: 800px) {

    .summary-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .metrics {
        grid-template-columns: repeat(2, 1fr);
    }

}

"""


# ---------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------

with gr.Blocks(
    title="TruthLens — Evidence-Based Claim Verification",
) as demo:

    with gr.Column(elem_id="main-container"):

        gr.HTML(
            """
            <div class="hero">

                <h1>TruthLens</h1>

                <p>
                    Evidence-Based Claim Verification for AI-Generated Text
                </p>

            </div>
            """
        )

        gr.Markdown(
            """
### How it works

TruthLens breaks an AI-generated answer into individual claims,
retrieves relevant evidence from a fixed Wikipedia evidence corpus,
and uses a Natural Language Inference (NLI) model to classify each claim as:

- **SUPPORTED** — retrieved evidence entails the claim
- **CONTRADICTED** — retrieved evidence contradicts the claim
- **UNKNOWN** — the available evidence is insufficient to support or contradict the claim

> **Important:** UNKNOWN does not mean false. The system is an
> evidence-based verification system, not a universal truth detector.
            """
        )

        with gr.Row():

            answer_input = gr.Textbox(
                label="AI-Generated Answer",
                placeholder=(
                    "Paste an AI-generated answer here...\n\n"
                    "Example:\n"
                    "Paris is the capital of France. "
                    "It is located in Germany."
                ),
                lines=10,
            )

        with gr.Row():

            verify_button = gr.Button(
                "🔎 Verify Answer",
                variant="primary",
            )

            clear_button = gr.ClearButton(
                components=[answer_input],
                value="Clear",
            )

        gr.Examples(
            examples=[
                [EXAMPLE_ANSWER],
                [
                    "The Earth revolves around the Sun. "
                    "The Sun revolves around the Earth."
                ],
                [
                    "Albert Einstein developed the theory of relativity. "
                    "He was born in Germany."
                ],
            ],
            inputs=answer_input,
            label="Example Inputs",
        )

        gr.Markdown("## Results")

        summary_output = gr.HTML()

        claims_output = gr.HTML()

        gr.Markdown(
            """
---

### System limitations

- The current evidence corpus is a development-scale subset of
  Wikipedia derived from FEVER development evidence pages.
- Retrieval quality depends on the available evidence corpus.
- The system may classify a claim as UNKNOWN when relevant evidence
  is missing.
- Sentence-level claim extraction is intentionally lightweight.
- Pronouns and cross-sentence references may require additional
  coreference resolution.
- NLI confidence is model confidence, not a probability that a claim
  is objectively true.
- The contradicted fraction should not be interpreted as a universal
  hallucination rate.

### Model components

**Semantic retrieval:** `all-MiniLM-L6-v2`

**Lexical retrieval:** BM25

**Hybrid retrieval:** Reciprocal Rank Fusion

**Verification:** `cross-encoder/nli-deberta-v3-base`

**Evidence source:** FEVER-derived Wikipedia corpus
            """
        )

        verify_button.click(
            fn=verify_answer,
            inputs=answer_input,
            outputs=[
                summary_output,
                claims_output,
            ],
        )


# ---------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(css=CSS)