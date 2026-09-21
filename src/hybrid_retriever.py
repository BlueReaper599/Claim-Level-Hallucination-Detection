import json
import pickle
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------
# Files
# -------------------------------------------------------------------

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
BM25_METADATA_FILE = BM25_DIR / "metadata.jsonl"

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

TOP_K = 10

# Number of candidates retrieved from each retriever
CANDIDATE_K = 50

# RRF smoothing constant
RRF_K = 60

DEVICE = "cuda"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_jsonl(filename):
    records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_metadata(filename):
    return load_jsonl(filename)


def get_gold_pages(record):

    pages = set()

    for evidence_group in record.get("evidence", []):

        for evidence_item in evidence_group:

            if len(evidence_item) >= 3:
                pages.add(evidence_item[2])

    return pages


def reciprocal_rank_fusion(rankings):
    """
    Combine ranked document lists using Reciprocal Rank Fusion.

    rankings:
        list of lists of document IDs

    returns:
        dictionary mapping document ID -> fused score
    """

    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(ranking, start=1):

            score = 1.0 / (RRF_K + rank)

            scores[doc_id] = scores.get(doc_id, 0.0) + score

    return scores


def build_models():

    print("Loading FAISS index...")

    semantic_index = faiss.read_index(
        str(SEMANTIC_INDEX_FILE)
    )

    print(f"Semantic vectors: {semantic_index.ntotal:,}")

    print("\nLoading BM25 index...")

    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    print("BM25 loaded.")

    print("\nLoading metadata...")

    semantic_metadata = load_metadata(
        SEMANTIC_METADATA_FILE
    )

    bm25_metadata = load_metadata(
        BM25_METADATA_FILE
    )

    print(
        f"Semantic metadata: {len(semantic_metadata):,}"
    )

    print(
        f"BM25 metadata:     {len(bm25_metadata):,}"
    )

    print("\nLoading MiniLM...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=DEVICE
    )

    return (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        model,
    )


def hybrid_search(
    claim,
    semantic_index,
    bm25,
    semantic_metadata,
    bm25_metadata,
    model,
):
    """
    Retrieve evidence using both semantic and lexical retrieval.
    """

    # ---------------------------------------------------------------
    # Semantic retrieval
    # ---------------------------------------------------------------

    query_embedding = model.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, semantic_indices = (
        semantic_index.search(
            query_embedding,
            CANDIDATE_K
        )
    )

    semantic_ranking = [
        int(idx)
        for idx in semantic_indices[0]
        if idx >= 0
    ]

    # ---------------------------------------------------------------
    # BM25 retrieval
    # ---------------------------------------------------------------

    query_tokens = tokenize(claim)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = bm25_scores.argsort()[
        -CANDIDATE_K:
    ][::-1]

    bm25_ranking = [
        int(idx)
        for idx in bm25_indices
    ]

    # ---------------------------------------------------------------
    # RRF fusion
    # ---------------------------------------------------------------

    fused_scores = reciprocal_rank_fusion(
        [
            semantic_ranking,
            bm25_ranking
        ]
    )

    # Sort documents by fused score.
    ranked_documents = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # ---------------------------------------------------------------
    # Return top K
    # ---------------------------------------------------------------

    results = []

    for doc_id, fused_score in ranked_documents[:TOP_K]:

        evidence = semantic_metadata[doc_id]

        results.append({
            "page_id": evidence["page_id"],
            "sentence_id": evidence["sentence_id"],
            "text": evidence["text"],
            "score": fused_score,
        })

    return results


def main():

    print("=" * 70)
    print("Hybrid Retrieval Test")
    print("=" * 70)

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        model,
    ) = build_models()

    dev_data = load_jsonl(DEV_FILE)

    # Test on first development example.
    record = dev_data[0]

    claim = record["claim"]

    gold_pages = get_gold_pages(record)

    print("\n" + "=" * 70)
    print("CLAIM")
    print("=" * 70)

    print(claim)

    print("\nGold pages:")

    for page in gold_pages:
        print(f"  - {page}")

    results = hybrid_search(
        claim,
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        model,
    )

    print("\n" + "=" * 70)
    print("HYBRID TOP RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\nRank {rank}")
        print(f"RRF score   : {result['score']:.6f}")
        print(f"Page        : {result['page_id']}")
        print(f"Sentence ID : {result['sentence_id']}")
        print(
            f"Gold page?  : "
            f"{result['page_id'] in gold_pages}"
        )
        print(f"Evidence    : {result['text']}")


if __name__ == "__main__":
    main()