from app.retrieval.pipeline import retrieve_with_reranking


MIN_RERANK_SCORE = -5.0
MIN_TERM_COVERAGE = 0.20


def _extract_query_terms(query):
    stopwords = {
        "a", "an", "and", "are", "as", "be", "by", "for",
        "from", "how", "in", "is", "it", "of", "on", "or",
        "that", "the", "this", "to", "was", "what", "when",
        "where", "which", "who", "why", "with", "does", "do",
        "can", "could", "would", "should", "is", "are"
    }

    words = query.lower().split()

    terms = []

    for word in words:
        word = word.strip(
            ".,?!:;()[]{}\"'"
        )

        if word and word not in stopwords:
            terms.append(word)

    return terms


def _calculate_term_coverage(
    query_terms,
    documents
):
    if not query_terms:
        return 1.0

    context = " ".join(
        document.page_content.lower()
        for document, _ in documents
    )

    matched = 0

    for term in query_terms:
        if term in context:
            matched += 1

    return matched / len(query_terms)


def check_grounding(
    query,
    documents,
    final_k=3,
    candidate_k=10,
    semantic_k=10,
    bm25_k=10,
    semantic_weight=0.5,
    bm25_weight=0.5
):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    results = retrieve_with_reranking(
        query=query,
        documents=documents,
        final_k=final_k,
        candidate_k=candidate_k,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight
    )

    if not results:

        return {
            "grounded": False,
            "reason": "No relevant documents were retrieved.",
            "results": []
        }

    best_score = results[0][1]

    if best_score < MIN_RERANK_SCORE:

        return {
            "grounded": False,
            "reason": (
                "Retrieved evidence is not "
                "sufficiently relevant."
            ),
            "results": results
        }

    query_terms = _extract_query_terms(query)

    term_coverage = _calculate_term_coverage(
        query_terms,
        results
    )

    if term_coverage < MIN_TERM_COVERAGE:

        return {
            "grounded": False,
            "reason": (
                "The retrieved documents do not "
                "contain enough of the requested "
                "query terms."
            ),
            "results": results
        }

    return {
        "grounded": True,
        "reason": (
            "Relevant evidence containing "
            "sufficient query terms was retrieved."
        ),
        "term_coverage": term_coverage,
        "results": results
    }