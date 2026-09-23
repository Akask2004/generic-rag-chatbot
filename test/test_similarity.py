import numpy as np

from app.embeddings import get_embeddings


def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    similarity = np.dot(
        vector_a,
        vector_b
    ) / (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    return similarity


def main():

    print("=" * 70)
    print("COSINE SIMILARITY TEST")
    print("=" * 70)

    embeddings = get_embeddings()

    texts = [
        "What is legal research?",
        "Explain legal research methodology.",
        "What is the capital of India?",
        "How does a neural network work?"
    ]

    vectors = [
        embeddings.embed_query(text)
        for text in texts
    ]

    # Compare first sentence with all others

    query = texts[0]
    query_vector = vectors[0]

    print("\nQuery:")
    print(query)

    print("\nSimilarity scores:")
    print("-" * 70)

    for text, vector in zip(
        texts[1:],
        vectors[1:]
    ):

        score = cosine_similarity(
            query_vector,
            vector
        )

        print(
            f"{score:.4f}  →  {text}"
        )


if __name__ == "__main__":
    main()
    