from app.retrieval.retrieval import bm25_search
from app.vectorstore import retrieve_documents


def normalize_scores(results):
    """
    Normalize retrieval scores between 0 and 1.

    Higher normalized score means a better result.
    """

    if not results:
        return []

    scores = [
        score
        for _, score in results
        if score is not None
    ]

    if not scores:
        return [
            (document, 0.0)
            for document, _ in results
        ]

    max_score = max(scores)

    min_score = min(scores)

    if max_score == min_score:

        return [
            (document, 1.0)
            for document, _ in results
        ]

    normalized = []

    for document, score in results:

        normalized_score = (
            (score - min_score)
            / (max_score - min_score)
        )

        normalized.append(
            (
                document,
                normalized_score
            )
        )

    return normalized


def hybrid_search(
    query,
    documents,
    k=3,
    semantic_k=10,
    bm25_k=10,
    semantic_weight=0.5,
    bm25_weight=0.5,
    metadata_filter=None
):
    """
    Combine semantic retrieval and BM25 retrieval.

    Parameters
    ----------
    query : str
        User query.

    documents : list
        Documents used by BM25.

    k : int
        Number of final results.

    semantic_k : int
        Number of semantic candidates.

    bm25_k : int
        Number of BM25 candidates.

    semantic_weight : float
        Weight given to semantic retrieval.

    bm25_weight : float
        Weight given to BM25 retrieval.

    metadata_filter : dict or None
        Optional metadata filter.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if k <= 0:

        raise ValueError(
            "k must be greater than 0."
        )

    if semantic_weight < 0:

        raise ValueError(
            "semantic_weight cannot be negative."
        )

    if bm25_weight < 0:

        raise ValueError(
            "bm25_weight cannot be negative."
        )

    total_weight = (
        semantic_weight
        + bm25_weight
    )

    if total_weight == 0:

        raise ValueError(
            "At least one retrieval weight "
            "must be greater than 0."
        )

    semantic_results = retrieve_documents(
        query=query,
        k=semantic_k,
        metadata_filter=metadata_filter
    )

    semantic_scores = []

    for document, distance in semantic_results:

        if distance is None:
            continue

        semantic_scores.append(
            (
                document,
                1.0 / (1.0 + distance)
            )
        )

    semantic_scores = normalize_scores(
        semantic_scores
    )

    bm25_results = bm25_search(
        query=query,
        documents=documents,
        k=bm25_k,
        metadata_filter=metadata_filter
    )

    bm25_scores = normalize_scores(
        bm25_results
    )

    combined_scores = {}

    documents_by_id = {}

    for document, score in semantic_scores:

        document_id = document.metadata.get(
            "chunk_id",
            str(id(document))
        )

        combined_scores[document_id] = (
            combined_scores.get(
                document_id,
                0.0
            )
            + semantic_weight * score
        )

        documents_by_id[document_id] = document

    for document, score in bm25_scores:

        document_id = document.metadata.get(
            "chunk_id",
            str(id(document))
        )

        combined_scores[document_id] = (
            combined_scores.get(
                document_id,
                0.0
            )
            + bm25_weight * score
        )

        documents_by_id[document_id] = document

    ranked_results = []

    for document_id, score in combined_scores.items():

        ranked_results.append(
            (
                documents_by_id[document_id],
                score / total_weight
            )
        )

    ranked_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results[:k]