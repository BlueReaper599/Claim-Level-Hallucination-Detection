import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, DEVICE


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "shared_task_dev.jsonl"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_index"
)

FAISS_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.jsonl"


TOP_K = 5


def load_metadata():
    records = []

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_dev_record(index=0):
    with open(DEV_FILE, "r", encoding="utf-8") as f:

        for i, line in enumerate(f):

            if not line.strip():
                continue

            if i == index:
                return json.loads(line)

    raise IndexError("Development record not found.")


def get_gold_pages(record):
    pages = set()

    for evidence_group in record.get("evidence", []):

        for evidence_item in evidence_group:

            if len(evidence_item) >= 3:
                pages.add(evidence_item[2])

    return pages


def main():

    print("=" * 70)
    print("Loading FAISS index")
    print("=" * 70)

    index = faiss.read_index(
        str(FAISS_FILE)
    )

    print(f"Indexed vectors: {index.ntotal:,}")

    print("\nLoading metadata...")

    metadata = load_metadata()

    print(f"Metadata records: {len(metadata):,}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE
    )

    # Use the first FEVER development example.
    record = load_dev_record(0)

    claim = record["claim"]

    gold_pages = get_gold_pages(record)

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)

    print(claim)

    print("\nGold evidence pages:")

    for page in gold_pages:
        print(f"  - {page}")

    # Encode claim.
    query_embedding = model.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search.
    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    print("\n" + "=" * 70)
    print(f"TOP {TOP_K} RETRIEVED EVIDENCE")
    print("=" * 70)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        evidence = metadata[idx]

        page_id = evidence["page_id"]
        sentence_id = evidence["sentence_id"]
        text = evidence["text"]

        is_gold_page = page_id in gold_pages

        print(f"\nRank {rank}")
        print(f"Score       : {score:.4f}")
        print(f"Page        : {page_id}")
        print(f"Sentence ID : {sentence_id}")
        print(f"Gold page?  : {is_gold_page}")
        print(f"Evidence    : {text}")


if __name__ == "__main__":
    main()