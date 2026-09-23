from app.vectorstore import load_vectorstore
from app.retrieval.retrieval import bm25_search


QUERIES = [
    "What are questionnaires and interview methods?",
    "What is bibliographical research?",
    "What is comparative research?",
]


def main():

    print("=" * 70)
    print("BM25 RETRIEVAL TEST")
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

        results = bm25_search(
            query,
            documents,
            k=3
        )

        if not results:

            print(
                "\nNo BM25 matches found."
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
                f"BM25 Score: {score:.4f}"
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