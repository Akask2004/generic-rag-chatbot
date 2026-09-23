def calculate_retrieval_success(results):
    """
    Check whether the system retrieved any sources.
    """

    if not results:
        return 0

    return 1


def calculate_source_coverage(result):
    """
    Check whether the final answer has supporting sources.
    """

    if not result:
        return 0

    if not result.get("answer"):
        return 0

    if not result.get("sources"):
        return 0

    return 1


def calculate_answer_length(answer):
    """
    Return the number of words in the answer.
    """

    if not answer:
        return 0

    return len(answer.split())


def calculate_evaluation_metrics(
    result,
    expected_type
):
    """
    Calculate basic evaluation metrics for one RAG result.
    """

    sources = result.get("sources", [])
    answer = result.get("answer", "")

    retrieval_success = calculate_retrieval_success(
        sources
    )

    source_coverage = calculate_source_coverage(
        result
    )

    answer_length = calculate_answer_length(
        answer
    )

    if expected_type == "unanswerable":

        correctly_refused = (
            len(sources) == 0
        )

    else:

        correctly_refused = (
            len(sources) == 0
        )

    return {
        "retrieval_success": retrieval_success,
        "source_coverage": source_coverage,
        "answer_length": answer_length,
        "correctly_refused": int(correctly_refused),
    }