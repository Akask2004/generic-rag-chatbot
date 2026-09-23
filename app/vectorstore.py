from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.ingestions.loader import load_documents
from app.ingestions.cleaner import clean_documents
from app.ingestions.chunker import split_documents
from app.ingestions.metadata import enrich_metadata
from app.embeddings import get_embeddings


VECTORSTORE_DIR = Path("data/vectorstorage")

_vectorstore = None


def create_vectorstore():

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No supported documents found in data/documents/"
        )

    print(
        f"Loaded {len(documents)} document sections."
    )

    documents = clean_documents(documents)

    print("Documents cleaned.")

    chunks = split_documents(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    chunks = enrich_metadata(chunks)

    print("Metadata enriched.")

    embeddings = get_embeddings()

    print("Creating embeddings...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    global _vectorstore

    _vectorstore = vectorstore

    print(
        "Vector store created successfully."
    )

    return vectorstore


def rebuild_vectorstore():

    global _vectorstore

    print(
        "Rebuilding vector store..."
    )

    _vectorstore = None

    return create_vectorstore()


def load_vectorstore():

    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    if not VECTORSTORE_DIR.exists():

        raise FileNotFoundError(
            "Vector store does not exist. "
            "Create it first."
        )

    print(
        "Loading vector store..."
    )

    embeddings = get_embeddings()

    _vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return _vectorstore


def retrieve_documents(
    query,
    k=3,
    score_threshold=None,
    search_type="similarity",
    metadata_filter=None
):

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    if (
        score_threshold is not None
        and score_threshold < 0
    ):
        raise ValueError(
            "score_threshold cannot be negative."
        )

    if search_type not in {
        "similarity",
        "mmr"
    }:
        raise ValueError(
            "search_type must be "
            "'similarity' or 'mmr'."
        )

    if (
        metadata_filter is not None
        and not isinstance(
            metadata_filter,
            dict
        )
    ):
        raise TypeError(
            "metadata_filter must be a dictionary."
        )

    vectorstore = load_vectorstore()

    if search_type == "similarity":

        results = (
            vectorstore
            .similarity_search_with_score(
                query,
                k=k,
                filter=metadata_filter
            )
        )

    else:

        documents = (
            vectorstore
            .max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=max(k * 3, 10),
                lambda_mult=0.5,
                filter=metadata_filter
            )
        )

        results = [
            (document, None)
            for document in documents
        ]

    if score_threshold is not None:

        results = [
            (document, score)
            for document, score in results
            if (
                score is not None
                and score <= score_threshold
            )
        ]

    return results


if __name__ == "__main__":
    create_vectorstore()