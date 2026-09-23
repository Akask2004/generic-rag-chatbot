from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        print("Loading reranker model...")

        _reranker = CrossEncoder(
            MODEL_NAME,
            device="cpu"
        )

    return _reranker


def rerank_documents(
    query,
    documents,
    top_k=3
):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not documents:
        return []

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    reranker = get_reranker()

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_results = list(
        zip(documents, scores)
    )

    ranked_results.sort(
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        (document, float(score))
        for document, score in ranked_results[:top_k]
    ]