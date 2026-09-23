from app.vectorstore import load_vectorstore


def main():

    print("=" * 70)
    print("VECTOR STORE TEST")
    print("=" * 70)

    vectorstore = load_vectorstore()

    print("\nVector store loaded successfully.")

    # Number of stored vectors
    total_vectors = (
        vectorstore.index.ntotal
    )

    print(
        f"Total vectors: {total_vectors}"
    )

    # Vector dimension
    dimension = (
        vectorstore.index.d
    )

    print(
        f"Vector dimension: {dimension}"
    )

    # Stored documents
    documents = (
        vectorstore.docstore._dict
    )

    print(
        f"Stored documents: {len(documents)}"
    )

    # Show first document
    print("\n")
    print("=" * 70)
    print("FIRST STORED CHUNK")
    print("=" * 70)

    first_document = next(
        iter(documents.values())
    )

    print(
        "\nContent:"
    )

    print(
        first_document.page_content
    )

    print(
        "\nMetadata:"
    )

    print(
        first_document.metadata
    )


if __name__ == "__main__":
    main()