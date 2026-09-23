from app.ingestions.loader import load_documents
from app.ingestions.chunker import split_documents


def main():

    print("=" * 70)
    print("TESTING DOCUMENT INGESTION")
    print("=" * 70)

    # Step 1: Load documents
    documents = load_documents()

    print("\n")
    print("=" * 70)
    print("RAW DOCUMENT")
    print("=" * 70)

    print(documents[0].page_content[:1000])

    # Step 2: Chunk documents
    chunks = split_documents(documents)

    print("\n")
    print("=" * 70)
    print("FIRST CHUNK")
    print("=" * 70)

    print(chunks[0].page_content)

    print("\n")
    print("=" * 70)
    print("METADATA")
    print("=" * 70)

    print(chunks[0].metadata)


if __name__ == "__main__":
    main()