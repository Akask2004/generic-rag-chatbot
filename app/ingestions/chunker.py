import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _is_heading(line):
    """
    Detect whether a line looks like a document heading.

    This function is intentionally generic and does not
    depend on any specific document type.
    """

    line = line.strip()

    if not line:
        return False

    # Markdown headings
    if re.match(r"^#{1,6}\s+", line):
        return True

    # Numbered headings:
    # 1. Introduction
    # 1.1 Background
    # 2.3.1 Methodology
    if re.match(
        r"^\d+(?:\.\d+)*\.?\s+[A-Z]",
        line
    ):
        return True

    # Roman numeral headings:
    # I. Introduction
    # II. Methodology
    if re.match(
        r"^[IVXLCDM]+\.?\s+[A-Z]",
        line,
        re.IGNORECASE
    ):
        return True

    # Common document section names
    common_headings = {
        "abstract",
        "introduction",
        "background",
        "overview",
        "methodology",
        "methods",
        "materials and methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "appendix",
        "summary",
    }

    if line.lower().rstrip(":") in common_headings:
        return True

    return False


def _split_into_sections(text):
    """
    Split text into sections using detected headings.

    Returns
    -------
    list
        List of dictionaries containing heading and content.
    """

    lines = text.splitlines()

    sections = []

    current_heading = None
    current_content = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if _is_heading(line):

            # Save previous section
            if current_heading is not None:

                sections.append(
                    {
                        "heading": current_heading,
                        "content": "\n".join(
                            current_content
                        ).strip()
                    }
                )

            current_heading = line
            current_content = []

        else:

            current_content.append(line)

    # Save final section
    if current_heading is not None:

        sections.append(
            {
                "heading": current_heading,
                "content": "\n".join(
                    current_content
                ).strip()
            }
        )

    # If no headings were detected,
    # return the entire document as one section.
    if not sections:

        return [
            {
                "heading": None,
                "content": text.strip()
            }
        ]

    return sections


def split_documents(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """
    Generic structure-aware document chunking.

    The function first detects common document
    headings and creates sections. Large sections
    are then recursively split according to the
    configured chunk size.

    Parameters
    ----------
    documents : list
        LangChain Document objects.

    chunk_size : int
        Maximum size of each chunk.

    chunk_overlap : int
        Number of overlapping characters between
        consecutive chunks.

    Returns
    -------
    list
        Chunked LangChain Documents.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    final_chunks = []

    for document in documents:

        text = document.page_content.strip()

        if not text:
            continue

        sections = _split_into_sections(text)

        for section in sections:

            heading = section["heading"]
            content = section["content"]

            if heading:

                section_text = (
                    heading
                    + "\n"
                    + content
                ).strip()

            else:

                section_text = content

            if not section_text:
                continue

            section_document = Document(
                page_content=section_text,
                metadata=document.metadata.copy()
            )

            # Store detected heading in metadata
            if heading:

                section_document.metadata[
                    "section"
                ] = heading

            chunks = text_splitter.split_documents(
                [section_document]
            )

            final_chunks.extend(chunks)

    return final_chunks