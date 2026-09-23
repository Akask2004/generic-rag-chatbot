from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.ingestions.loader import load_documents
from app.ingestions.cleaner import clean_documents
from app.ingestions.chunker import split_documents
from app.ingestions.metadata import enrich_metadata
from app.embeddings import get_embeddings


VECTORSTORE_DIR = Path(
    "data/vectorstorage"
)


def create_vectorstore():
    """
    Create a FAISS vector store from
    documents in the data directory.
    """

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No documents found in data/documents/"
        )

    print(
        f"Loaded {len(documents)} document pages."
    )

    documents = clean_documents(
        documents
    )

    print(
        "Documents cleaned."
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    chunks = enrich_metadata(
        chunks
    )

    print(
        "Metadata enriched."
    )

    embeddings = get_embeddings()

    print(
        "Creating embeddings..."
    )

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

    print(
        "Vector store created successfully."
    )

    return vectorstore


def load_vectorstore():
    """
    Load an existing FAISS vector database.
    """

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store does not exist. "
            "Create it first."
        )

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve_documents(
    query,
    k=3,
    score_threshold=None,
    search_type="similarity",
    metadata_filter=None
):
    """
    Retrieve relevant documents for a query.

    Parameters
    ----------
    query : str
        User's question.

    k : int
        Number of documents to retrieve.

    score_threshold : float or None
        Optional maximum FAISS distance.

    search_type : str
        "similarity" or "mmr".

    metadata_filter : dict or None
        Optional metadata filter.

        Example:
            {
                "document_name": "example.pdf"
            }

    Returns
    -------
    list
        List of (Document, score) tuples.
    """

    # --------------------------------
    # Validate query
    # --------------------------------

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    # --------------------------------
    # Validate k
    # --------------------------------

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    # --------------------------------
    # Validate threshold
    # --------------------------------

    if score_threshold is not None:

        if score_threshold < 0:
            raise ValueError(
                "score_threshold cannot be negative."
            )

    # --------------------------------
    # Validate search type
    # --------------------------------

    valid_search_types = {
        "similarity",
        "mmr"
    }

    if search_type not in valid_search_types:

        raise ValueError(
            "search_type must be "
            "'similarity' or 'mmr'."
        )

    # --------------------------------
    # Validate metadata filter
    # --------------------------------

    if metadata_filter is not None:

        if not isinstance(
            metadata_filter,
            dict
        ):

            raise TypeError(
                "metadata_filter must be "
                "a dictionary."
            )

    # --------------------------------
    # Load vector store
    # --------------------------------

    vectorstore = load_vectorstore()

    # --------------------------------
    # Similarity search
    # --------------------------------

    if search_type == "similarity":

        results = (
            vectorstore
            .similarity_search_with_score(
                query,
                k=k,
                filter=metadata_filter
            )
        )

    # --------------------------------
    # MMR search
    # --------------------------------

    else:

        documents = (
            vectorstore
            .max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=max(
                    k * 3,
                    10
                ),
                lambda_mult=0.5,
                filter=metadata_filter
            )
        )

        results = [
            (document, None)
            for document in documents
        ]

    # --------------------------------
    # Apply threshold
    # --------------------------------

    if score_threshold is not None:

        results = [
            (document, score)
            for document, score in results
            if score is not None
            and score <= score_threshold
        ]

    return results


if __name__ == "__main__":
    create_vectorstore()