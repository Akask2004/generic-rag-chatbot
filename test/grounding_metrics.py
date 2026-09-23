import re


def normalize_text(text):
    """
    Normalize text for simple comparison.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_answer_sentences(answer):
    """
    Split an answer into simple sentences.
    """

    if not answer:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def calculate_sentence_support(
    sentence,
    context
):
    """
    Estimate how much of a sentence
    is supported by the retrieved context.

    This is a simple lexical metric.
    It is not a semantic truth checker.
    """

    sentence_tokens = set(
        normalize_text(sentence).split()
    )

    context_tokens = set(
        normalize_text(context).split()
    )

    if not sentence_tokens:
        return 0.0

    matching_tokens = (
        sentence_tokens & context_tokens
    )

    return (
        len(matching_tokens)
        / len(sentence_tokens)
    )


def calculate_groundedness(
    answer,
    context,
    support_threshold=0.30
):
    """
    Calculate the percentage of answer
    sentences that have lexical support
    in the retrieved context.
    """

    if not answer:
        return {
            "groundedness": 0.0,
            "supported_sentences": 0,
            "total_sentences": 0,
        }

    if not context:
        return {
            "groundedness": 0.0,
            "supported_sentences": 0,
            "total_sentences": len(
                get_answer_sentences(answer)
            ),
        }

    sentences = get_answer_sentences(answer)

    supported_sentences = 0

    for sentence in sentences:

        support = calculate_sentence_support(
            sentence,
            context
        )

        if support >= support_threshold:
            supported_sentences += 1

    total_sentences = len(sentences)

    if total_sentences == 0:
        groundedness = 0.0
    else:
        groundedness = (
            supported_sentences
            / total_sentences
        )

    return {
        "groundedness": groundedness,
        "supported_sentences": supported_sentences,
        "total_sentences": total_sentences,
    }