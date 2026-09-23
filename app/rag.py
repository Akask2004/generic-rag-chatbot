from app.vectorstore import load_vectorstore
from app.grounding import check_grounding
from app.generator import generate_answer


def build_context_from_results(results):
    """
    Build the LLM context from reranked documents.
    """

    if not results:
        return "", []

    context_parts = []
    sources = []

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):
        page = document.metadata.get(
            "page_label",
            "N/A"
        )

        document_name = document.metadata.get(
            "document_name",
            "Unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "Unknown"
        )

        context_parts.append(
            f"[Source {rank}]\n"
            f"Document: {document_name}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

        sources.append(
            {
                "rank": rank,
                "score": score,
                "document": document_name,
                "page": page,
                "chunk_id": chunk_id
            }
        )

    context = "\n\n".join(context_parts)

    return context, sources


def build_context(
    query,
    final_k=3,
    candidate_k=10,
    semantic_k=10,
    bm25_k=10,
    semantic_weight=0.5,
    bm25_weight=0.5
):
    """
    Retrieve and rerank documents, then build
    context for the language model.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    vectorstore = load_vectorstore()

    documents = list(
        vectorstore.docstore._dict.values()
    )

    grounding_result = check_grounding(
        query=query,
        documents=documents,
        final_k=final_k,
        candidate_k=candidate_k,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight
    )

    results = grounding_result["results"]

    if not grounding_result["grounded"]:
        return {
            "grounded": False,
            "reason": grounding_result["reason"],
            "context": "",
            "sources": []
        }

    context, sources = build_context_from_results(
        results
    )

    return {
        "grounded": True,
        "reason": grounding_result["reason"],
        "context": context,
        "sources": sources
    }


def ask_rag(
    query,
    final_k=3,
    candidate_k=10,
    semantic_k=10,
    bm25_k=10,
    semantic_weight=0.5,
    bm25_weight=0.5
):
    """
    Complete RAG question-answering pipeline.
    """

    result = build_context(
        query=query,
        final_k=final_k,
        candidate_k=candidate_k,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight
    )

    if not result["grounded"]:
        return {
            "answer": (
                "I could not find enough relevant "
                "information in the provided documents "
                "to answer this question."
            ),
            "sources": []
        }

    answer = generate_answer(
        query=query,
        context=result["context"]
    )

    return {
        "answer": answer,
        "sources": result["sources"]
    }