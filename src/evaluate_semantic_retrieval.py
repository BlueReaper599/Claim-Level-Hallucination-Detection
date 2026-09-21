import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

TOP_K = 10


def load_metadata():
    records = []

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_dev_data():
    records = []

    with open(DEV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_gold_pages(record):
    """
    Extract Wikipedia page IDs that contain gold evidence.
    """

    pages = set()

    for evidence_group in record.get("evidence", []):

        for evidence_item in evidence_group:

            if len(evidence_item) >= 3:
                pages.add(evidence_item[2])

    return pages


def main():

    print("=" * 70)
    print("Semantic Retrieval Evaluation")
    print("=" * 70)

    print("\nLoading FAISS index...")

    index = faiss.read_index(
        str(FAISS_FILE)
    )

    print(f"Indexed vectors: {index.ntotal:,}")

    print("\nLoading metadata...")

    metadata = load_metadata()

    print(f"Metadata records: {len(metadata):,}")

    print("\nLoading FEVER development set...")

    dev_data = load_dev_data()

    print(f"Development records: {len(dev_data):,}")

    print("\nLoading MiniLM...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE
    )

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    evaluated = 0

    print("\nRunning retrieval evaluation...\n")

    for record in tqdm(dev_data):

        claim = record["claim"]

        gold_pages = get_gold_pages(record)

        # Some records may not have evidence.
        if not gold_pages:
            continue

        query_embedding = model.encode(
            [claim],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(
            query_embedding,
            TOP_K
        )

        retrieved_pages = [
            metadata[idx]["page_id"]
            for idx in indices[0]
            if idx >= 0
        ]

        retrieved_at_1 = set(retrieved_pages[:1])
        retrieved_at_5 = set(retrieved_pages[:5])
        retrieved_at_10 = set(retrieved_pages[:10])

        if gold_pages.intersection(retrieved_at_1):
            recall_at_1 += 1

        if gold_pages.intersection(retrieved_at_5):
            recall_at_5 += 1

        if gold_pages.intersection(retrieved_at_10):
            recall_at_10 += 1

        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No development examples were evaluated.")

    recall_at_1 /= evaluated
    recall_at_5 /= evaluated
    recall_at_10 /= evaluated

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Examples evaluated : {evaluated:,}")
    print(f"Recall@1           : {recall_at_1:.4f}")
    print(f"Recall@5           : {recall_at_5:.4f}")
    print(f"Recall@10          : {recall_at_10:.4f}")

    print("\nPercentage:")

    print(f"Recall@1  : {recall_at_1 * 100:.2f}%")
    print(f"Recall@5  : {recall_at_5 * 100:.2f}%")
    print(f"Recall@10 : {recall_at_10 * 100:.2f}%")


if __name__ == "__main__":
    main()