from app.vectorstore import retrieve_documents


def main():

    print("=" * 70)
    print("METADATA FILTER TEST")
    print("=" * 70)

    query = "What is doctrinal research?"

    print("\nQUERY:")
    print(query)

    # --------------------------------
    # Search without filter
    # --------------------------------

    print("\n" + "-" * 70)
    print("SEARCH WITHOUT FILTER")
    print("-" * 70)

    results = retrieve_documents(
        query,
        k=3
    )

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {rank}"
        )

        print(
            f"Document: "
            f"{document.metadata.get('document_name')}"
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
            f"Distance: "
            f"{score:.4f}"
        )

    # --------------------------------
    # Search with document filter
    # --------------------------------

    print("\n" + "-" * 70)
    print("SEARCH WITH DOCUMENT FILTER")
    print("-" * 70)

    results = retrieve_documents(
        query,
        k=3,
        metadata_filter={
            "document_name": "LLM -1  SYLLABUS.pdf"
        }
    )

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {rank}"
        )

        print(
            f"Document: "
            f"{document.metadata.get('document_name')}"
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
            f"Distance: "
            f"{score:.4f}"
        )


if __name__ == "__main__":
    main()