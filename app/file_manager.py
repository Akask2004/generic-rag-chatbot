from pathlib import Path
import shutil

from app.vectorstore import rebuild_vectorstore


DOCUMENTS_DIR = Path("data/documents")

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}


def initialize_documents_directory():
    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def validate_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Not a file: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return file_path


def get_unique_destination(file_path):
    """
    Generate a unique filename if the same
    filename already exists.
    """

    destination = DOCUMENTS_DIR / file_path.name

    if not destination.exists():
        return destination

    stem = file_path.stem
    suffix = file_path.suffix

    counter = 1

    while True:

        new_name = (
            f"{stem}_{counter}{suffix}"
        )

        destination = (
            DOCUMENTS_DIR / new_name
        )

        if not destination.exists():
            return destination

        counter += 1


def add_file(file_path):
    """
    Add one document and rebuild the vector store.
    """

    initialize_documents_directory()

    file_path = validate_file(file_path)

    destination = get_unique_destination(
        file_path
    )

    shutil.copy2(
        file_path,
        destination
    )

    print(
        f"Added file: {destination.name}"
    )

    rebuild_vectorstore()

    return destination


def add_files(file_paths):
    """
    Add multiple documents and rebuild once.
    """

    initialize_documents_directory()

    added_files = []

    for file_path in file_paths:

        file_path = validate_file(file_path)

        destination = get_unique_destination(
            file_path
        )

        shutil.copy2(
            file_path,
            destination
        )

        added_files.append(destination)

        print(
            f"Added file: {destination.name}"
        )

    if added_files:
        rebuild_vectorstore()

    return added_files


def remove_file(file_name):
    """
    Remove a document and rebuild the vector store.
    """

    initialize_documents_directory()

    file_path = (
        DOCUMENTS_DIR / file_name
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_name}"
        )

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: "
            f"{file_path.suffix}"
        )

    file_path.unlink()

    print(
        f"Removed file: {file_name}"
    )

    rebuild_vectorstore()


def list_files():
    """
    Return all supported documents.
    """

    initialize_documents_directory()

    return sorted(
        file
        for file in DOCUMENTS_DIR.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in ALLOWED_EXTENSIONS
        )
    )