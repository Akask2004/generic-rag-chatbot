from app.rag import ask_rag, build_context
from test.evaluation_questions import EVALUATION_QUESTIONS
from test.evaluation_metrics import calculate_evaluation_metrics
from test.grounding_metrics import calculate_groundedness


def main():
    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    total = len(EVALUATION_QUESTIONS)

    total_retrieval_success = 0
    total_source_coverage = 0
    total_correctly_refused = 0
    total_answer_length = 0
    total_groundedness = 0.0

    for index, item in enumerate(EVALUATION_QUESTIONS, start=1):

        question = item["question"]
        expected_type = item["expected_type"]

        print("\n" + "=" * 70)
        print(f"QUESTION {index}/{total}")
        print("=" * 70)

        print(f"Question: {question}")
        print(f"Expected type: {expected_type}")

        result = ask_rag(
            query=question,
            final_k=3,
            candidate_k=10,
            semantic_k=10,
            bm25_k=10,
            semantic_weight=0.5,
            bm25_weight=0.5
        )

        metrics = calculate_evaluation_metrics(
            result=result,
            expected_type=expected_type
        )

        # Get the same retrieved context used by RAG.
        context_result = build_context(
            query=question,
            final_k=3,
            candidate_k=10,
            semantic_k=10,
            bm25_k=10,
            semantic_weight=0.5,
            bm25_weight=0.5
        )

        grounding = calculate_groundedness(
            answer=result["answer"],
            context=context_result["context"]
        )

        total_retrieval_success += metrics[
            "retrieval_success"
        ]

        total_source_coverage += metrics[
            "source_coverage"
        ]

        total_correctly_refused += metrics[
            "correctly_refused"
        ]

        total_answer_length += metrics[
            "answer_length"
        ]

        total_groundedness += grounding[
            "groundedness"
        ]

        print("\nANSWER")
        print("-" * 70)
        print(result["answer"])

        print("\nSOURCES")
        print("-" * 70)

        if not result["sources"]:
            print("No sources returned.")

        else:
            for source in result["sources"]:
                print(
                    f"Rank {source['rank']} | "
                    f"Score {source['score']:.4f} | "
                    f"Page {source['page']} | "
                    f"{source['chunk_id']}"
                )

        print("\nMETRICS")
        print("-" * 70)

        print(
            f"Retrieval success: "
            f"{metrics['retrieval_success']}"
        )

        print(
            f"Source coverage: "
            f"{metrics['source_coverage']}"
        )

        print(
            f"Answer length: "
            f"{metrics['answer_length']} words"
        )

        print(
            f"Correctly refused: "
            f"{metrics['correctly_refused']}"
        )

        print(
            f"Groundedness: "
            f"{grounding['groundedness']:.2%}"
        )

        print(
            f"Supported sentences: "
            f"{grounding['supported_sentences']}/"
            f"{grounding['total_sentences']}"
        )

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION")
    print("=" * 70)

    retrieval_rate = (
        total_retrieval_success / total
    ) * 100

    source_coverage_rate = (
        total_source_coverage / total
    ) * 100

    refusal_rate = (
        total_correctly_refused / total
    ) * 100

    average_answer_length = (
        total_answer_length / total
    )

    average_groundedness = (
        total_groundedness / total
    ) * 100

    print(
        f"Retrieval success rate: "
        f"{retrieval_rate:.2f}%"
    )

    print(
        f"Source coverage rate: "
        f"{source_coverage_rate:.2f}%"
    )

    print(
        f"Correct refusal rate: "
        f"{refusal_rate:.2f}%"
    )

    print(
        f"Average answer length: "
        f"{average_answer_length:.2f} words"
    )

    print(
        f"Average groundedness: "
        f"{average_groundedness:.2f}%"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()