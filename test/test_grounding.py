from app.vectorstore import load_vectorstore
from app.grounding import check_grounding


QUERIES = [
    "What is doctrinal research?",
    "What is bibliographical research?",
    "What is the capital of France?",
    "What are questionnaires and interview methods?",
]


def main():
    print("=" * 70)
    print("GROUNDING TEST")
    print("=" * 70)

    vectorstore = load_vectorstore()

    documents = list(vectorstore.docstore._dict.values())

    for query in QUERIES:

        print("\n" + "=" * 70)
        print(f"QUESTION: {query}")
        print("=" * 70)

        result = check_grounding(
            query=query,
            documents=documents,
            final_k=3,
            candidate_k=10,
            semantic_k=10,
            bm25_k=10,
            semantic_weight=0.5,
            bm25_weight=0.5
        )

        print("\nGROUNDED:")
        print(result["grounded"])

        print("\nREASON:")
        print(result["reason"])

        print("\nRETRIEVED RESULTS:")
        print("-" * 70)

        for rank, (document, score) in enumerate(
            result["results"],
            start=1
        ):
            print(f"Rank: {rank}")
            print(f"Score: {score:.4f}")
            print(
                f"Page: "
                f"{document.metadata.get('page_label', 'N/A')}"
            )
            print(
                f"Chunk: "
                f"{document.metadata.get('chunk_id', 'Unknown')}"
            )
            print()


if __name__ == "__main__":
    main()