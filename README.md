---
title: Generic RAG Chatbot
emoji: R
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
python_version: 3.11
---

# Generic RAG Chatbot

A generic Retrieval-Augmented Generation chatbot that allows users to upload PDF, DOCX, and TXT documents and ask questions based only on the uploaded content.

## Features

- PDF, DOCX and TXT support
- Document cleaning
- Adaptive chunking
- Metadata enrichment
- Hugging Face embeddings
- FAISS vector search
- BM25 keyword retrieval
- Hybrid retrieval
- Cross-encoder reranking
- Grounding checks
- Source/page references
- Local open-source LLM generation
- Streamlit interface

## Architecture

Documents  
↓  
Loader  
↓  
Cleaner  
↓  
Chunker  
↓  
Metadata  
↓  
Embeddings  
↓  
FAISS + BM25  
↓  
Hybrid Retrieval  
↓  
Reranking  
↓  
Grounding Check  
↓  
Qwen LLM  
↓  
Answer + Sources


# Local RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built with Python.

The chatbot can read PDF, TXT, and DOCX documents, create local embeddings, store them in FAISS, retrieve relevant information, and generate answers using a local LLM through Ollama.

## Features

* PDF, TXT, and DOCX document support
* Local text embeddings using Hugging Face
* FAISS vector database
* Local LLM using Ollama
* No OpenAI API key required
* Simple command-line chatbot
* Answers based on uploaded documents

## Project Structure

```text
rag_chatbot/
│
├── data/
│   ├── documents/
│   │   └── your_documents_here
│   │
│   └── vectorstore/
│
├── app/
│   ├── __init__.py
│   ├── loader.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── rag.py
│   └── main.py
│
├── .env
├── requirements.txt
└── README.md
```

## 1. Create Virtual Environment

From the project directory:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

## 2. Install Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## 3. Install Ollama

Install Ollama on your computer.

After installation, download the Llama model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before starting the chatbot.

## 4. Add Documents

Place your documents inside:

```text
data/documents/
```

Example:

```text
data/
└── documents/
    ├── sample.pdf
    ├── notes.txt
    └── report.docx
```

## 5. Run the Chatbot

From the project root:

```bash
python -m app.main
```

The first run will:

1. Load the documents
2. Split them into chunks
3. Generate embeddings
4. Create the FAISS vector database
5. Start the chatbot

## 6. Chat With Your Documents

Example:

```text
==================================================
       LOCAL RAG CHATBOT
==================================================

Creating vector database...

RAG chatbot is ready!
Type 'exit' to quit.

You: What is this document about?

Bot: The document discusses...
```

To exit:

```text
You: exit
```

## How It Works

The application follows this RAG pipeline:

```text
Documents
    │
    ▼
loader.py
    │
    ▼
Text Chunks
    │
    ▼
embeddings.py
    │
    ▼
Hugging Face Embeddings
    │
    ▼
FAISS Vector Database
    │
    ▼
User Question
    │
    ▼
Retriever
    │
    ▼
Relevant Document Chunks
    │
    ▼
Ollama / Llama
    │
    ▼
Generated Answer
```

## Technologies

| Technology            | Purpose                 |
| --------------------- | ----------------------- |
| Python                | Application development |
| LangChain             | RAG pipeline            |
| Hugging Face          | Local embeddings        |
| Sentence Transformers | Embedding model         |
| FAISS                 | Vector database         |
| Ollama                | Local LLM               |
| Llama 3.2             | Text generation         |
| PyPDF                 | PDF loading             |

## Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model runs locally and does not require an API key.

## LLM

The project uses:

```text
llama3.2
```

through Ollama.

The model runs locally, so the chatbot does not need OpenAI or another paid LLM API.

## Important Notes

The first time the embedding model is used, it will be downloaded to your computer.

The first time you create the vector database, embeddings will be generated for all documents.

If you add or modify documents after creating the vector database, delete:

```text
data/vectorstore/
```

and run the application again.

This will rebuild the vector database.

## Future Improvements

Possible improvements include:

* Streamlit web interface
* Chat history
* Source citations
* Multiple-document management
* Better chunking strategies
* Reranking
* Conversation memory
* Document upload interface
* Persistent chat sessions
* Support for more file formats
* Local model selection
* REST API using Flask or FastAPI

## License

This project is intended for learning and experimentation.
