from hybrid_retriever import (
    build_models,
    hybrid_search,
)


def main():

    (
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    ) = build_models()

    claim = "Paris is the capital of France."

    results = hybrid_search(
        claim,
        semantic_index,
        bm25,
        semantic_metadata,
        bm25_metadata,
        embedding_model,
    )

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL OUTPUT")
    print("=" * 80)

    for i, result in enumerate(
        results[:10],
        start=1
    ):

        print(f"\nResult {i}")
        print("-" * 80)

        print(result)


if __name__ == "__main__":
    main()