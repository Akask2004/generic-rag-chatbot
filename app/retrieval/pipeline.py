from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank_documents


def retrieve_with_reranking(
    query,
    documents,
    final_k=3,
    candidate_k=10,
    semantic_k=10,
    bm25_k=10,
    semantic_weight=0.5,
    bm25_weight=0.5,
    metadata_filter=None
):
    """
    Perform hybrid retrieval followed by reranking.

    Pipeline:

        Semantic Retrieval
                +
           BM25 Retrieval
                |
                v
        Hybrid Candidates
                |
                v
          Cross-Encoder
                |
                v
        Final Documents
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if final_k <= 0:

        raise ValueError(
            "final_k must be greater than 0."
        )

    if candidate_k <= 0:

        raise ValueError(
            "candidate_k must be greater than 0."
        )

    hybrid_results = hybrid_search(
        query=query,
        documents=documents,
        k=candidate_k,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        metadata_filter=metadata_filter
    )

    if not hybrid_results:

        return []

    candidate_documents = [
        document
        for document, _ in hybrid_results
    ]

    reranked_results = rerank_documents(
        query=query,
        documents=candidate_documents,
        top_k=final_k
    )

    return reranked_results