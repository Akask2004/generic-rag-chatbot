from app.vectorstore import load_vectorstore


def main():

    vectorstore = load_vectorstore()

    query = "What is doctrinal research?"

    print("=" * 70)
    print("TOP-K RETRIEVAL EXPERIMENT")
    print("=" * 70)

    print(f"\nQuery: {query}")

    for k in [1, 2, 3, 5]:

        print("\n")
        print("=" * 70)
        print(f"TOP K = {k}")
        print("=" * 70)

        results = (
            vectorstore
            .similarity_search_with_score(
                query,
                k=k
            )
        )

        for rank, (document, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"\n{rank}. "
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
                "Content preview:"
            )

            print(
                document.page_content[:250]
                .replace("\n", " ")
            )


if __name__ == "__main__":
    main()