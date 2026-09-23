from app.ingestions.loader import load_documents
from app.ingestions.cleaner import clean_documents
from app.ingestions.chunker import split_documents


def main():

    documents = load_documents()

    documents = clean_documents(
        documents
    )

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=50
    )

    print("=" * 70)
    print("CHUNKING TEST")
    print("=" * 70)

    print(
        f"\nDocuments: {len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    for i, chunk in enumerate(
        chunks[:5]
    ):

        print("\n")
        print("-" * 70)
        print(f"CHUNK {i + 1}")
        print("-" * 70)

        print(
            chunk.page_content
        )

        print("\nMetadata:")

        print(
            chunk.metadata
        )


if __name__ == "__main__":
    main()