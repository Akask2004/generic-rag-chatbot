from app.vectorstore import load_vectorstore


EVALUATION_DATA = [
    {
        "question": "What is doctrinal research?",
        "expected_terms": ["doctrinal research", "doctrinal"]
    },
    {
        "question": "What is empirical research?",
        "expected_terms": ["empirical research"]
    },
    {
        "question": "What is a research problem?",
        "expected_terms": ["research problem"]
    },
    {
        "question": "What is research design?",
        "expected_terms": ["research design"]
    },
    {
        "question": "What are questionnaires and interviews?",
        "expected_terms": ["questionnaires", "interview"]
    }
]


def contains_expected_term(
    text,
    expected_terms
):
    """
    Check whether the retrieved text
    contains at least one expected term.
    """

    text = text.lower()

    return any(
        term.lower() in text
        for term in expected_terms
    )


def evaluate_retrieval(
    vectorstore,
    k
):

    correct = 0

    for item in EVALUATION_DATA:

        results = (
            vectorstore
            .similarity_search_with_score(
                item["question"],
                k=k
            )
        )

        found = False

        for document, score in results:

            if contains_expected_term(
                document.page_content,
                item["expected_terms"]
            ):
                found = True
                break

        if found:
            correct += 1

    accuracy = (
        correct / len(EVALUATION_DATA)
    ) * 100

    return accuracy


def main():

    print("=" * 70)
    print("CONTENT-BASED RETRIEVAL EVALUATION")
    print("=" * 70)

    vectorstore = load_vectorstore()

    for k in [1, 3, 5]:

        accuracy = evaluate_retrieval(
            vectorstore,
            k
        )

        print(
            f"\nTop-{k} Accuracy: "
            f"{accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()