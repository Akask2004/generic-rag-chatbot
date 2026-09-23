from app.vectorstore import load_vectorstore
from app.retrieval.hybrid import hybrid_search


QUERIES = [
    "What is doctrinal research?",
    "What is empirical research?",
    "What are questionnaires and interview methods?",
    "What is bibliographical research?",
    "What is comparative research?",
]


def main():

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    vectorstore = load_vectorstore()

    documents = list(
        vectorstore.docstore._dict.values()
    )

    print(
        f"\nTotal documents/chunks: "
        f"{len(documents)}"
    )

    for query in QUERIES:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = hybrid_search(
            query=query,
            documents=documents,
            k=3,
            semantic_k=10,
            bm25_k=10,
            semantic_weight=0.5,
            bm25_weight=0.5
        )

        if not results:

            print(
                "\nNo hybrid results found."
            )

            continue

        for rank, (document, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRESULT {rank}"
            )

            print(
                f"Hybrid Score: {score:.4f}"
            )

            print(
                f"Document: "
                f"{document.metadata.get('document_name', 'N/A')}"
            )

            print(
                f"Page: "
                f"{document.metadata.get('page_label', 'N/A')}"
            )

            print(
                f"Chunk ID: "
                f"{document.metadata.get('chunk_id', 'N/A')}"
            )

            print("\nContent:")

            print(
                document.page_content
            )


if __name__ == "__main__":
    main()