from app.vectorstore import load_vectorstore


def main():

    print("=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    vectorstore = load_vectorstore()

    queries = [
        "What is doctrinal research?",
        "What is empirical research?",
        "What is a research problem?"
    ]

    for query in queries:

        print("\n")
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = (
            vectorstore
            .similarity_search_with_score(
                query,
                k=3
            )
        )

        for rank, (document, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRESULT {rank}"
            )

            print(
                f"Distance: {score:.4f}"
            )

            print(
                f"Page: "
                f"{document.metadata.get('page_label')}"
            )

            print(
                f"Chunk ID: "
                f"{document.metadata.get('chunk_id')}"
            )

            print(
                "\nContent:"
            )

            print(
                document.page_content
            )


if __name__ == "__main__":
    main()