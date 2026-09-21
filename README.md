# TruthLens: Claim-Level Hallucination Detection Using Evidence Retrieval and NLI

An evidence-based system for detecting potentially hallucinated claims in AI-generated answers.

The system decomposes an AI-generated answer into individual claims, retrieves relevant evidence from a fixed Wikipedia-derived corpus, and uses Natural Language Inference (NLI) to classify each claim as:

* **SUPPORTED** — retrieved evidence supports the claim.
* **CONTRADICTED** — retrieved evidence contradicts the claim.
* **UNKNOWN** — the available evidence is insufficient to make a reliable decision.

> TruthLens is an evidence-based factuality verification system, not a universal truth detector. In particular, absence of evidence is not treated as proof that a claim is false.

---

## 1. Problem

Large Language Models can generate fluent statements that are unsupported or factually incorrect.

A simple answer-level classifier can hide this problem because an answer may contain several individually correct and incorrect statements.

TruthLens therefore performs **claim-level verification**.

For example:

```text
AI-generated answer
        ↓
Claim extraction
        ↓
Individual claims
        ↓
Evidence retrieval
        ↓
NLI verification
        ↓
SUPPORTED / CONTRADICTED / UNKNOWN
```

---

## 2. Research Question

The main research question is:

> Does combining lexical retrieval and semantic retrieval with Natural Language Inference improve claim-level hallucination detection compared with individual retrieval approaches?

The project also investigates how the number of retrieved evidence sentences affects NLI-based verification.

---

## 3. System Architecture

```text
                    AI-generated answer
                            │
                            ▼
                    Claim extraction
                            │
                            ▼
                    Individual claims
                            │
                            ▼
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
          BM25                           MiniLM
      lexical search                semantic search
             │                             │
             └──────────────┬──────────────┘
                            ▼
                   Reciprocal Rank Fusion
                            │
                            ▼
                    Top evidence sentence
                            │
                            ▼
                    DeBERTa-v3 NLI model
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
             Entailment  Contradiction Neutral
                 │          │          │
                 ▼          ▼          ▼
             SUPPORTED  CONTRADICTED  UNKNOWN
```

---

## 4. Main Components

### Claim extraction

The current implementation uses lightweight sentence-level segmentation.

This provides a simple and reproducible claim extraction baseline.

### BM25 retrieval

BM25 provides lexical retrieval based on word overlap.

It is useful for retrieving evidence containing important terms from the claim.

### Semantic retrieval

The system uses:

`sentence-transformers/all-MiniLM-L6-v2`

to generate dense sentence embeddings.

The embeddings are indexed using FAISS.

### Hybrid retrieval

BM25 and semantic retrieval are combined using **Reciprocal Rank Fusion (RRF)**.

The hybrid retriever is used to produce the final evidence ranking.

### NLI verification

The system uses:

`cross-encoder/nli-deberta-v3-base`

The NLI labels are mapped as:

```text
entailment    -> SUPPORTED
contradiction -> CONTRADICTED
neutral       -> UNKNOWN
```

A confidence threshold and decision-margin threshold are used to avoid forcing uncertain cases into supported or contradicted classes.

---

## 5. Datasets

### FEVER

The primary benchmark is the **FEVER (Fact Extraction and VERification)** dataset.

Official sources:

* FEVER dataset and documentation: https://fever.ai/
* Official FEVER repository: https://github.com/awslabs/fever

The project uses:

* FEVER training data (`train.jsonl`)
* FEVER development data (`shared_task_dev.jsonl`)
* FEVER Wikipedia evidence pages (`wiki-pages.zip`)

The development set contains claims labelled:

```text
SUPPORTS
REFUTES
NOT ENOUGH INFO
```

For the project pipeline, these are mapped to:

```text
SUPPORTS          -> SUPPORTED
REFUTES           -> CONTRADICTED
NOT ENOUGH INFO   -> UNKNOWN
```

The experiments use a development-scale evidence corpus constructed from the Wikipedia pages referenced by the FEVER development evidence.

The resulting evidence corpus contains:

**27,055 sentences**

The complete FEVER Wikipedia corpus is much larger and is therefore not loaded into RAM during development experiments.

The raw FEVER files and generated evidence/index files are intentionally excluded from GitHub because of their size. They can be reconstructed locally using the scripts in `src/`.

### HaluEval

HaluEval is used as a secondary external benchmark for testing transfer beyond FEVER-style claim verification.

Official source:

* HaluEval repository: https://github.com/RUCAIBox/HaluEval

The current experiment evaluates the zero-shot NLI formulation on a **1,000-example subset** of the HaluEval QA data.

This experiment showed limited transfer of the FEVER-oriented NLI formulation to the QA setting. This is treated as a documented limitation rather than evidence that the complete hallucination-detection pipeline fails on HaluEval.

Future work includes QA-specific claim extraction, evidence retrieval, and verification.

---

## 6. Retrieval Results

The retrieval experiments were performed on **19,998 FEVER development examples** using a restricted development evidence corpus.

### Page-level retrieval

| Retriever            | Recall@1 | Recall@5 | Recall@10 |
| -------------------- | -------: | -------: | --------: |
| BM25                 |   50.72% |   62.69% |    64.37% |
| MiniLM semantic      |   55.69% |   64.22% |    65.11% |
| Hybrid BM25 + MiniLM |   56.51% |   65.02% |    65.67% |

### Sentence-level retrieval

On a 100-example subset, 74 examples contained usable sentence-level gold evidence.

| Retriever       | Recall@1 | Recall@5 | Recall@10 |
| --------------- | -------: | -------: | --------: |
| BM25            |   59.46% |   78.38% |    83.78% |
| MiniLM semantic |   58.11% |   74.32% |    81.08% |
| Hybrid          |   63.51% |   82.43% |    87.84% |

The sentence-level experiment should be interpreted separately from the page-level experiment because the evaluation populations and gold-evidence definitions differ.

---

## 7. NLI / Verification Ablation

A 1,000-example development subset was used to compare verification strategies.

| System              | Accuracy | Macro F1 | Weighted F1 |
| ------------------- | -------: | -------: | ----------: |
| BM25 + NLI + V2     |   47.40% |   44.82% |      44.95% |
| Semantic + NLI + V2 |   49.30% |   48.27% |      48.38% |
| Hybrid + NLI + V2   |   50.70% |   49.60% |      49.71% |
| Hybrid + NLI + V1   |   47.40% |   37.36% |      37.57% |

A second ablation compared the number of evidence sentences supplied to the NLI model:

| Configuration           | Accuracy | Macro F1 |
| ----------------------- | -------: | -------: |
| Hybrid + Top-1 NLI      |   60.80% |   61.00% |
| Hybrid + Top-5 NLI + V1 |   47.40% |   37.36% |
| Hybrid + Top-5 NLI + V2 |   50.70% |   49.60% |

The experiment indicates that supplying multiple retrieved sentences to the current aggregation strategy can dilute a strong relevant signal with neutral or irrelevant evidence.

Therefore, the frozen final architecture uses the **top-ranked evidence sentence** for NLI verification.

---

## 8. Held-Out Evaluation

After architecture selection on the first 1,000 FEVER development examples, a separate **2,000-example subset** was used as a held-out evaluation.

The architecture was frozen before this evaluation.

### Results

```text
Accuracy       : 62.25%
Macro Precision: 64.40%
Macro Recall   : 62.18%
Macro F1       : 62.31%
Weighted F1    : 62.46%
```

Class-level results:

| Class        | Precision | Recall |   F1 |
| ------------ | --------: | -----: | ---: |
| SUPPORTED    |      0.82 |   0.58 | 0.68 |
| CONTRADICTED |      0.63 |   0.77 | 0.69 |
| UNKNOWN      |      0.48 |   0.52 | 0.50 |

Confusion matrix:

```text
                 Predicted

               S      C      U

Actual S      393     58    229
       C       15    514    140
       U       69    244    338
```

These results are specific to the evaluated FEVER development subset and restricted evidence corpus. They should not be interpreted as general performance on all possible real-world LLM outputs.

---

## 9. Error Analysis

A 100-example error analysis produced:

```text
Correct                         57
Retrieval miss                   6
NLI / verification error        28
Unknown error type               9
```

The analysis suggests that retrieval is not the only bottleneck.

Even when relevant evidence is retrieved, the verification and decision stage can make incorrect or overly conservative decisions.

This motivates future work on:

* evidence reranking
* better evidence aggregation
* claim decomposition
* coreference resolution
* domain-specific verification
* calibrated confidence estimation

---

## 10. External HaluEval Experiment

A 1,000-example subset of HaluEval QA data was evaluated using a question-aware NLI formulation:

```text
Premise:

Knowledge + Question

Hypothesis:

Candidate answer
```

The resulting ROC-AUC was approximately:

**0.5065**

This is close to random discrimination.

The result indicates that the FEVER-oriented zero-shot NLI formulation does not transfer directly to the tested HaluEval QA setting.

This is treated as a limitation and negative-transfer finding rather than being hidden from the project results.

---

## 11. TruthLens Dashboard

The project includes a Gradio-based interface.

The dashboard accepts an AI-generated answer and displays:

* extracted claims
* final classification
* confidence
* decision margin
* NLI probabilities
* retrieved evidence
* Wikipedia page
* sentence identifier
* retrieval score

Run:

```powershell
python app/app.py
```

Then open:

```text
http://127.0.0.1:7860
```

---

## 12. Installation

### Requirements

Recommended environment:

```text
Python 3.11
NVIDIA GPU recommended
```

The project was developed and tested using an NVIDIA RTX 4060 Laptop GPU.

Create and activate the environment:

```powershell
conda create -n hallucination-detector python=3.11
conda activate hallucination-detector
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

For the exact development environment, see:

```text
requirements-lock.txt
```

---

## 13. Project Structure

```text
Claim-Level-Hallucination-Detection/

├── app/
│   └── app.py

├── data/
│   ├── raw/
│   └── processed/

├── evaluation/
│   └── results/

├── notebooks/

├── src/
│   ├── config.py
│   ├── dataset_loader.py
│   ├── preprocess.py
│   ├── build_evidence_corpus.py
│   ├── build_dev_corpus.py
│   ├── build_semantic_index.py
│   ├── build_bm25_index.py
│   ├── hybrid_retriever.py
│   ├── nli_verifier.py
│   ├── claim_verifier.py
│   ├── pipeline.py
│   └── ...

├── tests/
│   ├── test_pipeline.py
│   └── test_integration.py

├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

---

## 14. Testing

The project contains unit and integration tests.

Run the complete test suite:

```powershell
python -m pytest -v
```

Current status:

```text
13 tests passed
```

The integration tests verify the complete retrieval -> NLI -> decision pipeline.

---

## 15. Limitations

The current implementation has several important limitations.

### Restricted evidence corpus

The development retrieval experiments use a 27,055-sentence corpus constructed from Wikipedia pages referenced by FEVER development evidence.

This is not unrestricted full-Wikipedia retrieval.

### Claim extraction

Claim extraction currently uses lightweight sentence segmentation.

Complex claims containing multiple factual propositions may therefore require better decomposition.

### Coreference

Statements such as:

```text
Paris is the capital of France.

It is located in Europe.
```

may require coreference resolution for robust verification.

### NLI confidence

The NLI probability is a model output and should not be interpreted as an objective probability that a claim is true.

### UNKNOWN

UNKNOWN means that the current evidence and verifier did not provide sufficient support for a confident supported/contradicted decision.

It does not mean that the claim is necessarily false.

### Hallucination rate

The dashboard's contradicted fraction is the fraction of processed claims classified as CONTRADICTED.

It is not a universally validated hallucination rate.

### External benchmark transfer

The HaluEval experiment demonstrates that the current FEVER-oriented zero-shot NLI formulation does not directly transfer to the tested QA setting.

---

## 16. Ethical Considerations

A factuality verification system can itself make mistakes.

Potential risks include:

* incorrectly flagging factual statements
* failing to detect hallucinated statements
* over-reliance on retrieved evidence
* bias in the underlying evidence corpus
* misleading users through apparently precise confidence values

For this reason, TruthLens presents evidence and uncertainty rather than claiming to provide absolute truth.

The system is intended as an **assistive verification tool**, not an autonomous authority.

---

## 17. Future Work

Potential extensions include:

1. Full-Wikipedia retrieval.
2. Better claim decomposition.
3. Coreference resolution.
4. Evidence reranking using cross-encoders.
5. Fine-tuning NLI on factual verification data.
6. Calibration of confidence scores.
7. QA-specific verification for HaluEval.
8. Multi-hop evidence reasoning.
9. FEVEROUS evaluation.
10. Live web evidence retrieval with source provenance.

---

## 18. References

### Datasets

* FEVER: Fact Extraction and VERification
  https://fever.ai/

* FEVER official repository
  https://github.com/awslabs/fever

* HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models
  https://github.com/RUCAIBox/HaluEval

### Models

* `sentence-transformers/all-MiniLM-L6-v2`
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

* `cross-encoder/nli-deberta-v3-base`
  https://huggingface.co/cross-encoder/nli-deberta-v3-base

### Tools and Methods

* FAISS: Facebook AI Similarity Search
  https://github.com/facebookresearch/faiss

* BM25 information retrieval
  https://en.wikipedia.org/wiki/Okapi_BM25

* Sentence Transformers
  https://www.sbert.net/

* Hugging Face Transformers
  https://huggingface.co/docs/transformers/

* Gradio
  https://www.gradio.app/

---

## 19. License

Add the project license before publishing the repository.
