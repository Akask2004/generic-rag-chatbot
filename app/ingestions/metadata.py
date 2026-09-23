from pathlib import Path


def enrich_metadata(chunks):
    """
    Add useful and deterministic metadata
    to every chunk.
    """

    page_chunk_counters = {}

    for chunk in chunks:

        source = chunk.metadata.get(
            "source",
            "unknown"
        )

        source_path = Path(source)

        page = chunk.metadata.get(
            "page",
            0
        )

        page_label = chunk.metadata.get(
            "page_label",
            page + 1
        )

        # --------------------------------
        # Document information
        # --------------------------------

        chunk.metadata["document_name"] = (
            source_path.name
        )

        chunk.metadata["document_type"] = (
            source_path.suffix.lower().replace(
                ".",
                ""
            )
            or "unknown"
        )

        # --------------------------------
        # Page chunk counter
        # --------------------------------

        page_key = (
            f"{source_path.name}_"
            f"page_{page_label}"
        )

        if page_key not in page_chunk_counters:

            page_chunk_counters[page_key] = 0

        page_chunk_index = (
            page_chunk_counters[page_key]
        )

        page_chunk_counters[page_key] += 1

        # --------------------------------
        # Chunk information
        # --------------------------------

        chunk.metadata["chunk_index"] = (
            page_chunk_index
        )

        chunk.metadata["chunk_id"] = (
            f"{source_path.stem}"
            f"_page_{page_label}"
            f"_chunk_{page_chunk_index}"
        )

    return chunks