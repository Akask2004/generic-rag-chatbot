import re

from collections import Counter

from rank_bm25 import BM25Okapi


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with"
}


def tokenize(text):
    """
    Convert text into lowercase word tokens.
    """

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text
    )

    return tokens


def remove_stopwords(tokens):
    """
    Remove common words that usually
    carry little retrieval value.
    """

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def calculate_keyword_score(
    query,
    document
):
    """
    Calculate a simple keyword-overlap score.
    """

    query_tokens = tokenize(query)

    query_tokens = remove_stopwords(
        query_tokens
    )

    document_tokens = tokenize(
        document.page_content
    )

    if not query_tokens:
        return 0.0

    if not document_tokens:
        return 0.0

    document_counts = Counter(
        document_tokens
    )

    score = 0.0

    for token in query_tokens:

        if token in document_counts:

            score += document_counts[token]

    return score


def keyword_search(
    query,
    documents,
    k=3,
    metadata_filter=None
):
    """
    Search documents using simple keyword matching.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if k <= 0:

        raise ValueError(
            "k must be greater than 0."
        )

    if metadata_filter is not None:

        if not isinstance(
            metadata_filter,
            dict
        ):

            raise TypeError(
                "metadata_filter must be "
                "a dictionary."
            )

    scored_documents = []

    for document in documents:

        if metadata_filter:

            matches_filter = all(
                document.metadata.get(key) == value
                for key, value
                in metadata_filter.items()
            )

            if not matches_filter:
                continue

        score = calculate_keyword_score(
            query,
            document
        )

        if score > 0:

            scored_documents.append(
                (
                    document,
                    score
                )
            )

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return scored_documents[:k]


def bm25_search(
    query,
    documents,
    k=3,
    metadata_filter=None
):
    """
    Search documents using BM25 lexical retrieval.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if k <= 0:

        raise ValueError(
            "k must be greater than 0."
        )

    if metadata_filter is not None:

        if not isinstance(
            metadata_filter,
            dict
        ):

            raise TypeError(
                "metadata_filter must be "
                "a dictionary."
            )

    filtered_documents = []

    for document in documents:

        if metadata_filter:

            matches_filter = all(
                document.metadata.get(key) == value
                for key, value
                in metadata_filter.items()
            )

            if not matches_filter:
                continue

        filtered_documents.append(
            document
        )

    if not filtered_documents:
        return []

    tokenized_documents = []

    for document in filtered_documents:

        tokens = tokenize(
            document.page_content
        )

        tokens = remove_stopwords(
            tokens
        )

        tokenized_documents.append(
            tokens
        )

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = tokenize(
        query
    )

    query_tokens = remove_stopwords(
        query_tokens
    )

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_results = list(
        zip(
            filtered_documents,
            scores
        )
    )

    ranked_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for document, score in ranked_results[:k]:

        if score <= 0:
            continue

        results.append(
            (
                document,
                float(score)
            )
        )

    return results