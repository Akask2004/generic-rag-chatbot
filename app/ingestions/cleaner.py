import re

from langchain_core.documents import Document

def clean_text(text):
    """
    Clean extracted document text while
    preserving meaningful structure.
    """
    # --------------------------------
    # Basic whitespace cleaning
    # --------------------------------

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove lines containing only bullet symbols
    text = re.sub(
        r"(?m)^\s*[•▪◦]\s*$",
        "",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Strip whitespace from every line
    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    # --------------------------------
    # Remove obvious duplicate lines
    # --------------------------------

    cleaned_lines = []
    previous_line = None

    for line in lines:

        if not line:
            continue

        # Normalize the line only for comparison.
        # The original line is preserved.
        comparison_line = line.lower()

        # Remove trailing page/reference numbers
        # when comparing duplicate lines.
        comparison_line = re.sub(
            r"\s+\d+$",
            "",
            comparison_line
        )

        comparison_line = comparison_line.strip()

        if comparison_line == previous_line:
            continue

        cleaned_lines.append(line)

        previous_line = comparison_line

    # --------------------------------
    # Final cleanup
    # --------------------------------

    text = "\n".join(
        cleaned_lines
    ).strip()

    return text


def clean_documents(documents):
    """
    Clean the page_content of LangChain Documents.
    """
    cleaned_documents = []

    for document in documents:

        cleaned_content = clean_text(
            document.page_content
        )

        if not cleaned_content:
            continue

        cleaned_documents.append(
            Document(
                page_content=cleaned_content,
                metadata=document.metadata.copy()
            )
        )

    return cleaned_documents
