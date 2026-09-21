import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, DEVICE


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_evidence_corpus.jsonl"
)

INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "semantic_index"

EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE = INDEX_DIR / "metadata.jsonl"
FAISS_FILE = INDEX_DIR / "index.faiss"

BATCH_SIZE = 128


def load_corpus():
    """Load evidence sentences from the development corpus."""

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def build_index():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Loading development evidence corpus")
    print("=" * 70)

    records = load_corpus()

    print(f"Evidence sentences : {len(records):,}")

    texts = [record["text"] for record in records]

    print("\n" + "=" * 70)
    print("Loading embedding model")
    print("=" * 70)

    print(f"Model  : {EMBEDDING_MODEL}")
    print(f"Device : {DEVICE}")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE
    )

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    print("\nEmbedding shape:", embeddings.shape)

    # Save embeddings
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Saved embeddings: {EMBEDDINGS_FILE}")

    # Build FAISS index.
    # Because embeddings are normalized, inner product is equivalent
    # to cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS vectors: {index.ntotal:,}")

    faiss.write_index(
        index,
        str(FAISS_FILE)
    )

    print(f"Saved FAISS index: {FAISS_FILE}")

    # Save metadata separately so FAISS vector IDs can be mapped
    # back to the original evidence sentences.
    with open(METADATA_FILE, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved metadata: {METADATA_FILE}")

    print("\n" + "=" * 70)
    print("Semantic retrieval index built successfully!")
    print("=" * 70)

    print(f"Vectors   : {index.ntotal:,}")
    print(f"Dimension : {dimension}")
    print(f"Device    : {DEVICE}")


if __name__ == "__main__":
    build_index()