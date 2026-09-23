from app.rag import build_context


QUERIES = [
    "What is doctrinal research?",
    "What are questionnaires and interview methods?",
    "What is bibliographical research?",
]


def main():

    print("=" * 70)
    print("CONTEXT BUILDER TEST")
    print("=" * 70)

    for query in QUERIES:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        context, sources = build_context(
            query=query,
            final_k=3,
            candidate_k=10,
            semantic_k=10,
            bm25_k=10,
            semantic_weight=0.5,
            bm25_weight=0.5
        )

        if not context:

            print("\nNo context found.")

            continue

        print("\nCONTEXT")
        print("-" * 70)

        print(context)

        print("\nSOURCES")
        print("-" * 70)

        for source in sources:

            print(
                f"Rank: {source['rank']}"
            )

            print(
                f"Score: {source['score']:.4f}"
            )

            print(
                f"Document: {source['document']}"
            )

            print(
                f"Page: {source['page']}"
            )

            print(
                f"Chunk ID: {source['chunk_id']}"
            )

            print()


if __name__ == "__main__":
    main()