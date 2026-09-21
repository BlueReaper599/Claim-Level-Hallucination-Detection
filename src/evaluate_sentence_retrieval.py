import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ====================================================================
# PATHS
# ====================================================================

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

SEMANTIC_INDEX_FILE = (
    SEMANTIC_DIR
    / "index.faiss"
)

SEMANTIC_METADATA_FILE = (
    SEMANTIC_DIR
    / "metadata.jsonl"
)

BM25_FILE = (
    BM25_DIR
    / "bm25.pkl"
)

BM25_METADATA_FILE = (
    BM25_DIR
    / "metadata.jsonl"
)


# ====================================================================
# PARAMETERS
# ====================================================================

CANDIDATE_K = 50

RRF_K = 60

EVALUATION_LIMIT = 100


# ====================================================================
# DEVICE
# ====================================================================

import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ====================================================================
# BASIC HELPERS
# ====================================================================

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


def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ====================================================================
# GOLD EVIDENCE
# ====================================================================

def get_gold_sentences(record):

    """
    Extract the exact FEVER gold evidence
    as (page_id, sentence_id) pairs.
    """

    gold = set()

    for evidence_group in record.get(
        "evidence",
        []
    ):

        for evidence_item in evidence_group:

            if len(evidence_item) >= 4:

                page_id = evidence_item[2]

                sentence_id = evidence_item[3]

                # Ignore missing evidence placeholders.
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


# ====================================================================
# LOAD MODELS
# ====================================================================

def load_models():

    print("=" * 80)
    print("Loading retrieval resources")
    print("=" * 80)

    print("\nLoading FAISS index...")

    semantic_index = faiss.read_index(
        str(SEMANTIC_INDEX_FILE)
    )

    print(
        f"Semantic vectors: "
        f"{semantic_index.ntotal:,}"
    )

    print("\nLoading BM25 index...")

    with open(
        BM25_FILE,
        "rb"
    ) as f:

        bm25 = json.load(f) if False else __import__(
            "pickle"
        ).load(f)

    print("BM25 loaded.")

    print("\nLoading metadata...")

    semantic_metadata = load_jsonl(
        SEMANTIC_METADATA_FILE
    )

    bm25_metadata = load_jsonl(
        BM25_METADATA_FILE
    )

    print(
        f"Semantic metadata: "
        f"{len(semantic_metadata):,}"
    )

    print(
        f"BM25 metadata:     "
        f"{len(bm25_metadata):,}"
    )

    print("\nLoading MiniLM...")

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=DEVICE
    )

    return (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    )


# ====================================================================
# BM25 RETRIEVAL
# ====================================================================

def bm25_search(
    claim,
    bm25,
    metadata,
    top_k
):

    tokens = tokenize(claim)

    scores = bm25.get_scores(tokens)

    indices = np.argsort(scores)[
        -top_k:
    ][::-1]

    results = []

    for index in indices:

        index = int(index)

        results.append(
            metadata[index]
        )

    return results


# ====================================================================
# SEMANTIC RETRIEVAL
# ====================================================================

def semantic_search(
    claim,
    semantic_index,
    metadata,
    embedding_model,
    top_k
):

    query_embedding = embedding_model.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = (
        semantic_index.search(
            query_embedding,
            top_k
        )
    )

    results = []

    for index in indices[0]:

        index = int(index)

        if index >= 0:

            results.append(
                metadata[index]
            )

    return results


# ====================================================================
# HYBRID RETRIEVAL
# ====================================================================

def reciprocal_rank_fusion(
    rankings
):

    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):

            score = (
                1.0
                / (RRF_K + rank)
            )

            scores[doc_id] = (
                scores.get(
                    doc_id,
                    0.0
                )
                + score
            )

    return scores


def hybrid_search(
    claim,
    semantic_index,
    bm25,
    semantic_metadata,
    bm25_metadata,
    embedding_model,
    top_k
):

    semantic_results = semantic_search(
        claim,
        semantic_index,
        semantic_metadata,
        embedding_model,
        CANDIDATE_K
    )

    bm25_results = bm25_search(
        claim,
        bm25,
        bm25_metadata,
        CANDIDATE_K
    )

    # Use the metadata index as the document identity.
    #
    # Both metadata files were generated from the same corpus,
    # so page_id + sentence_id uniquely identifies an evidence
    # sentence.

    semantic_ids = []

    for result in semantic_results:

        semantic_ids.append(
            (
                result["page_id"],
                str(result["sentence_id"])
            )
        )

    bm25_ids = []

    for result in bm25_results:

        bm25_ids.append(
            (
                result["page_id"],
                str(result["sentence_id"])
            )
        )

    fused_scores = reciprocal_rank_fusion(
        [
            semantic_ids,
            bm25_ids
        ]
    )

    ranked_ids = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    metadata_lookup = {}

    for result in semantic_metadata:

        key = (
            result["page_id"],
            str(result["sentence_id"])
        )

        metadata_lookup[key] = result

    results = []

    for sentence_id, score in ranked_ids[
        :top_k
    ]:

        if sentence_id in metadata_lookup:

            result = dict(
                metadata_lookup[
                    sentence_id
                ]
            )

            result["rrf_score"] = score

            results.append(
                result
            )

    return results


# ====================================================================
# CHECK WHETHER GOLD EVIDENCE WAS RETRIEVED
# ====================================================================

def hit_at_k(
    results,
    gold_sentences,
    k
):

    retrieved = set()

    for result in results[:k]:

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


# ====================================================================
# MAIN
# ====================================================================

def main():

    print("=" * 80)
    print("SENTENCE-LEVEL RETRIEVAL EVALUATION")
    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Loading FEVER development data")
    print("=" * 80)

    dev_data = load_jsonl(
        DEV_FILE
    )

    print(
        f"Total examples: "
        f"{len(dev_data):,}"
    )

    if EVALUATION_LIMIT is None:

        evaluation_data = dev_data

    else:

        evaluation_data = dev_data[
            :EVALUATION_LIMIT
        ]

    print(
        f"Examples evaluated: "
        f"{len(evaluation_data):,}"
    )

    # ---------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = load_models()

    # ---------------------------------------------------------------
    # Counters
    # ---------------------------------------------------------------

    methods = [
        "BM25",
        "SEMANTIC",
        "HYBRID",
    ]

    hit_counts = {

        method: {
            1: 0,
            5: 0,
            10: 0,
        }

        for method in methods
    }

    evaluated = 0

    # ---------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Running sentence-level evaluation")
    print("=" * 80)

    for i, record in enumerate(
        evaluation_data,
        start=1
    ):

        gold_sentences = get_gold_sentences(
            record
        )

        # Some FEVER records may have no usable
        # sentence-level evidence.

        if not gold_sentences:

            continue

        claim = record["claim"]

        # -----------------------------------------------------------
        # BM25
        # -----------------------------------------------------------

        bm25_results = bm25_search(
            claim,
            bm25,
            bm25_metadata,
            CANDIDATE_K
        )

        # -----------------------------------------------------------
        # Semantic
        # -----------------------------------------------------------

        semantic_results = semantic_search(
            claim,
            semantic_index,
            semantic_metadata,
            embedding_model,
            CANDIDATE_K
        )

        # -----------------------------------------------------------
        # Hybrid
        # -----------------------------------------------------------

        hybrid_results = hybrid_search(
            claim,
            semantic_index,
            bm25,
            semantic_metadata,
            bm25_metadata,
            embedding_model,
            CANDIDATE_K
        )

        # -----------------------------------------------------------
        # Evaluate
        # -----------------------------------------------------------

        result_sets = {

            "BM25": bm25_results,

            "SEMANTIC": semantic_results,

            "HYBRID": hybrid_results,
        }

        for method, results in result_sets.items():

            for k in [1, 5, 10]:

                if hit_at_k(
                    results,
                    gold_sentences,
                    k
                ):

                    hit_counts[
                        method
                    ][k] += 1

        evaluated += 1

        if (
            i <= 10
            or i % 10 == 0
            or i == len(evaluation_data)
        ):

            print(
                f"[{i:>5}/"
                f"{len(evaluation_data)}] "
                f"evaluated={evaluated}"
            )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SENTENCE-LEVEL RETRIEVAL RESULTS")
    print("=" * 80)

    print(
        f"\nEvaluation examples with "
        f"sentence-level gold evidence: "
        f"{evaluated:,}"
    )

    print()

    print(
        f"{'Method':<12}"
        f"{'Recall@1':>15}"
        f"{'Recall@5':>15}"
        f"{'Recall@10':>15}"
    )

    print("-" * 57)

    for method in methods:

        r1 = (
            hit_counts[method][1]
            / evaluated
        )

        r5 = (
            hit_counts[method][5]
            / evaluated
        )

        r10 = (
            hit_counts[method][10]
            / evaluated
        )

        print(
            f"{method:<12}"
            f"{r1:>14.2%}"
            f"{r5:>14.2%}"
            f"{r10:>14.2%}"
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()