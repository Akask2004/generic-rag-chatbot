from app.ingestions.loader import load_documents
from app.ingestions.cleaner import clean_documents


def main():

    print("=" * 70)
    print("TEXT CLEANING TEST")
    print("=" * 70)

    # Load raw documents
    documents = load_documents()

    # Clean documents
    cleaned_documents = clean_documents(documents)

    # -----------------------------
    # Raw text
    # -----------------------------

    print("\n")
    print("=" * 70)
    print("RAW TEXT")
    print("=" * 70)

    print(
        documents[0].page_content[:1500]
    )

    # -----------------------------
    # Cleaned text
    # -----------------------------

    print("\n")
    print("=" * 70)
    print("CLEANED TEXT")
    print("=" * 70)

    print(
        cleaned_documents[0].page_content[:1500]
    )

    # -----------------------------
    # Metadata
    # -----------------------------

    print("\n")
    print("=" * 70)
    print("METADATA")
    print("=" * 70)

    print(
        cleaned_documents[0].metadata
    )


if __name__ == "__main__":
    main()