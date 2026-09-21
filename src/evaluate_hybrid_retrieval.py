import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "shared_task_dev.jsonl"
)

SEMANTIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_index"
)

BM25_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bm25_index"
)

SEMANTIC_INDEX_FILE = SEMANTIC_DIR / "index.faiss"
SEMANTIC_METADATA_FILE = SEMANTIC_DIR / "metadata.jsonl"

BM25_FILE = BM25_DIR / "bm25.pkl"

TOP_KS = [1, 5, 10]
CANDIDATE_K = 50
RRF_K = 60

DEVICE = "cuda"


def load_jsonl(filename):
    records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_gold_pages(record):

    pages = set()

    for evidence_group in record.get("evidence", []):

        for evidence_item in evidence_group:

            if len(evidence_item) >= 3:
                pages.add(evidence_item[2])

    return pages


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(rankings):

    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):

            score = 1.0 / (RRF_K + rank)

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + score
            )

    return scores


def main():

    print("=" * 70)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print("\nLoading development data...")

    dev_data = load_jsonl(DEV_FILE)

    print(
        f"Development records: {len(dev_data):,}"
    )

    # ---------------------------------------------------------------
    # Load FAISS
    # ---------------------------------------------------------------

    print("\nLoading semantic index...")

    semantic_index = faiss.read_index(
        str(SEMANTIC_INDEX_FILE)
    )

    semantic_metadata = load_jsonl(
        SEMANTIC_METADATA_FILE
    )

    print(
        f"Semantic documents: "
        f"{semantic_index.ntotal:,}"
    )

    # ---------------------------------------------------------------
    # Load BM25
    # ---------------------------------------------------------------

    print("\nLoading BM25...")

    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    print("BM25 loaded.")

    # ---------------------------------------------------------------
    # Load MiniLM
    # ---------------------------------------------------------------

    print("\nLoading MiniLM...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=DEVICE
    )

    # ---------------------------------------------------------------
    # Counters
    # ---------------------------------------------------------------

    hits = {
        1: 0,
        5: 0,
        10: 0,
    }

    evaluated = 0

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    print("\nStarting evaluation...")

    for record in tqdm(
        dev_data,
        desc="Hybrid retrieval"
    ):

        gold_pages = get_gold_pages(record)

        # Skip examples without annotated evidence.
        if not gold_pages:
            continue

        claim = record["claim"]

        # ===========================================================
        # Semantic retrieval
        # ===========================================================

        embedding = model.encode(
            [claim],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        _, semantic_indices = (
            semantic_index.search(
                embedding,
                CANDIDATE_K
            )
        )

        semantic_ranking = [
            int(idx)
            for idx in semantic_indices[0]
            if idx >= 0
        ]

        # ===========================================================
        # BM25 retrieval
        # ===========================================================

        query_tokens = tokenize(claim)

        bm25_scores = bm25.get_scores(
            query_tokens
        )

        bm25_indices = bm25_scores.argsort()[
            -CANDIDATE_K:
        ][::-1]

        bm25_ranking = [
            int(idx)
            for idx in bm25_indices
        ]

        # ===========================================================
        # RRF
        # ===========================================================

        fused_scores = reciprocal_rank_fusion(
            [
                semantic_ranking,
                bm25_ranking,
            ]
        )

        hybrid_ranking = [
            doc_id
            for doc_id, _ in sorted(
                fused_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # ===========================================================
        # Evaluate
        # ===========================================================

        retrieved_pages = [
            semantic_metadata[idx]["page_id"]
            for idx in hybrid_ranking
        ]

        evaluated += 1

        for k in TOP_KS:

            top_pages = set(
                retrieved_pages[:k]
            )

            if gold_pages & top_pages:

                hits[k] += 1

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nEvaluated examples: "
        f"{evaluated:,}"
    )

    print(
        "\nPage-level Recall:"
    )

    for k in TOP_KS:

        recall = hits[k] / evaluated

        print(
            f"Recall@{k}: "
            f"{recall:.4f} "
            f"({recall * 100:.2f}%)"
        )

    print("\nComparison with previous experiments:")

    print(
        "\n"
        "Retriever          Recall@1   Recall@5   Recall@10"
    )

    print(
        "BM25               50.72%     62.69%     64.37%"
    )

    print(
        "MiniLM             55.69%     64.22%     65.11%"
    )

    print(
        "Hybrid             "
        f"{hits[1] / evaluated * 100:.2f}%     "
        f"{hits[5] / evaluated * 100:.2f}%     "
        f"{hits[10] / evaluated * 100:.2f}%"
    )


if __name__ == "__main__":
    main()