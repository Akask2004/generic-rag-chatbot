from pathlib import Path

from app.rag import create_rag_chat, ask_question
from app.vectorstore import create_vectorstorage


VECTORSTORE_DIR = Path("data/vectorstorage")
FAISS_INDEX = VECTORSTORE_DIR / "index.faiss"


def main():

    print("=" * 50)
    print("       LOCAL RAG CHATBOT")
    print("=" * 50)

    # Check whether the actual FAISS index exists
    if not FAISS_INDEX.exists():

        print("\nFAISS vector database not found.")
        print("Creating vector database...\n")

        create_vectorstorage()

    else:

        print("\nExisting vector database found.")

    # Create RAG pipeline
    retriever, llm = create_rag_chat()

    print("\nRAG chatbot is ready!")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:

            answer = ask_question(
                question,
                retriever,
                llm
            )

            print(f"\nBot: {answer}\n")

        except Exception as e:

            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()