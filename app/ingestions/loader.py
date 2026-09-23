from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


DOCUMENTS_DIR = Path("data/documents")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}


def load_single_file(file_path):
    """
    Load one supported document.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        loader = PyMuPDFLoader(
            str(file_path)
        )

    elif extension == ".txt":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    elif extension == ".docx":

        loader = Docx2txtLoader(
            str(file_path)
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    return documents


def load_documents(
    documents_dir=DOCUMENTS_DIR
):
    """
    Load all supported documents
    from the documents directory.
    """

    documents_dir = Path(documents_dir)

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: "
            f"{documents_dir}"
        )

    all_documents = []

    files = sorted(
        file
        for file in documents_dir.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(
            "No supported documents found "
            "in data/documents/"
        )

        return []

    for file_path in files:

        print(
            f"Loading: {file_path.name}"
        )

        try:

            documents = load_single_file(
                file_path
            )

            all_documents.extend(
                documents
            )

            print(
                f"Loaded {len(documents)} "
                f"document sections."
            )

        except Exception as error:

            print(
                f"Failed to load "
                f"{file_path.name}: {error}"
            )

    return all_documents