from app.embeddings import get_embeddings


def main():

    print("=" * 70)
    print("EMBEDDING TEST")
    print("=" * 70)

    embeddings = get_embeddings()

    texts = [
        "What is legal research?",
        "Explain legal research methodology.",
        "What is the capital of India?",
        "How does a neural network work?"
    ]

    vectors = []

    for text in texts:

        vector = embeddings.embed_query(text)

        vectors.append(vector)

        print("\n")
        print("-" * 70)
        print("TEXT:")
        print(text)

        print(
            "Vector dimensions:",
            len(vector)
        )

        print(
            "First 5 values:",
            vector[:5]
        )

    print("\n")
    print("=" * 70)
    print("EMBEDDING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()