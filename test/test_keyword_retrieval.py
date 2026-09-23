from app.vectorstore import load_vectorstore
from app.retrieval.retrieval import keyword_search


QUERIES = [
    "What is doctrinal research?",
    "What is empirical research?",
    "What is a research problem?",
]


def main():

    print("=" * 70)
    print("KEYWORD RETRIEVAL TEST")
    print("=" * 70)

    # --------------------------------
    # Load existing vector store
    # --------------------------------

    vectorstore = load_vectorstore()

    # Get documents stored inside FAISS
    documents = list(
        vectorstore.docstore._dict.values()
    )

    print(
        f"\nTotal documents/chunks: "
        f"{len(documents)}"
    )

    # --------------------------------
    # Test queries
    # --------------------------------

    for query in QUERIES:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = keyword_search(
            query,
            documents,
            k=3
        )

        if not results:

            print(
                "\nNo keyword matches found."
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
                f"Keyword Score: {score:.4f}"
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