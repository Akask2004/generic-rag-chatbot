from app.rag import ask_rag


QUERIES = [
    "What types of research are mentioned?",
    "What is doctrinal research?",
    "What are questionnaires and interview methods?",
    "What is bibliographical research?",
]


def main():

    print("=" * 70)
    print("COMPLETE RAG TEST")
    print("=" * 70)

    for query in QUERIES:

        print("\n" + "=" * 70)
        print(f"QUESTION: {query}")
        print("=" * 70)

        result = ask_rag(
            query=query,
            final_k=3,
            candidate_k=10,
            semantic_k=10,
            bm25_k=10,
            semantic_weight=0.5,
            bm25_weight=0.5
        )

        print("\nANSWER")
        print("-" * 70)

        print(
            result["answer"]
        )

        print("\nSOURCES")
        print("-" * 70)

        for source in result["sources"]:

            print(
                f"Source {source['rank']}"
            )

            print(
                f"Score: "
                f"{source['score']:.4f}"
            )

            print(
                f"Document: "
                f"{source['document']}"
            )

            print(
                f"Page: "
                f"{source['page']}"
            )

            print(
                f"Chunk ID: "
                f"{source['chunk_id']}"
            )

            print()


if __name__ == "__main__":
    main()