import streamlit as st

from app.file_manager import add_files, list_files
from app.rag import ask_rag


st.set_page_config(
    page_title="Generic RAG Chatbot",
    layout="wide",
)

st.title("Generic RAG Chatbot")

st.write(
    "Upload documents and ask questions using only "
    "the information contained in those documents."
)

# --------------------------------------------------
# Upload Documents
# --------------------------------------------------

st.header("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process Documents"):

    temporary_files = []

    try:
        for uploaded_file in uploaded_files:

            temporary_path = (
                "data/"
                + uploaded_file.name
            )

            with open(temporary_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            temporary_files.append(temporary_path)

        added_files = add_files(temporary_files)

        st.success(
            f"Processed {len(added_files)} document(s)."
        )

    except Exception as error:
        st.error(f"Error processing documents: {error}")


# --------------------------------------------------
# Documents
# --------------------------------------------------

st.header("Documents")

documents = list_files()

if documents:

    for document in documents:
        st.write(f"- {document.name}")

else:
    st.info("No documents uploaded yet.")


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

st.header("Ask a Question")

query = st.text_input(
    "Enter your question"
)

if st.button("Ask"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents..."):

            try:

                result = ask_rag(
                    query=query
                )

                st.subheader("Answer")

                st.write(
                    result["answer"]
                )

                st.subheader("Sources")

                if result["sources"]:

                    for source in result["sources"]:

                        st.write(
                            f"**{source['document']}** "
                            f"| Page {source['page']} "
                            f"| {source['chunk_id']}"
                        )

                else:

                    st.write(
                        "No supporting sources found."
                    )

            except Exception as error:

                st.error(
                    f"Error: {error}"
                )