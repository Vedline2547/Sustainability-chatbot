# 🌱 Sustainability Coach – AI-Powered RAG Application

## Overview

Sustainability Coach is an AI-powered question-answering application that uses Retrieval-Augmented Generation (RAG) to help users understand sustainability-related information from PDF documents.

Built with Flask, LangChain, Ollama, and FAISS, the application processes uploaded PDF documents, converts their content into semantic chunks, generates vector embeddings, and retrieves relevant information to answer user questions.

The application provides context-aware responses and references the original PDF documents, making it easier for users to locate supporting information.

## Features

- **PDF Document Processing:** Extracts text from PDF documents using PDFPlumber.
- **Semantic Chunking:** Uses SemanticChunker to divide documents into meaningful text segments.
- **Text Embeddings:** Generates embeddings using Hugging Face sentence-transformer models.
- **Vector Database:** Stores and retrieves document embeddings using FAISS.
- **Retrieval-Augmented Generation:** Combines retrieved document context with an LLM to generate relevant answers.
- **Local LLM Integration:** Uses Qwen 2.5 (3B) through Ollama.
- **Source References:** Returns PDF references with page numbers for supporting information.
- **Web Interface:** Provides an interactive interface built with Flask.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Web application and backend |
| LangChain | RAG pipeline orchestration |
| Ollama | Local LLM inference |
| Qwen 2.5 (3B) | Language model |
| Hugging Face | Text embeddings |
| FAISS | Vector storage and similarity search |
| PDFPlumber | PDF text extraction |
| SemanticChunker | Semantic document splitting |

## Project Architecture

The application follows this workflow:

1. **Document Loading:** PDF documents are loaded from the `static/uploads` directory.
2. **Text Extraction:** PDFPlumber extracts the document content.
3. **Semantic Chunking:** Extracted text is divided into meaningful chunks.
4. **Embedding Generation:** Hugging Face generates vector representations of the text.
5. **Vector Storage:** FAISS stores the embeddings for efficient similarity search.
6. **Information Retrieval:** Relevant document chunks are retrieved based on the user's question.
7. **Answer Generation:** Qwen 2.5 receives the retrieved context and generates an answer.
8. **Source Referencing:** The application returns the relevant PDF document and page reference.

## Project Structure

```text
Sustainability-coach/
│
├── app.py
│
├── static/
│   └── uploads/
│       └── sustainability_documents.pdf
│
├── templates/
│   ├── landing.html
│   └── chat.html
│
├── faiss_index/
│
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Sustainability-chatbot.git

cd Sustainability-coach
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install flask langchain langchain-classic langchain-community langchain-experimental langchain-ollama sentence-transformers faiss-cpu pdfplumber
```

### 4. Install Ollama

Download Ollama from:

https://ollama.com/

Pull the Qwen 2.5 model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before starting the application.

### 5. Add PDF documents

Place your sustainability-related PDF documents inside:

```text
static/uploads/
```

### 6. Run the application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

## Example Use Case

A user uploads sustainability-related documentation and asks:

**Question:**

"What are the essential requirements concerning the reusable and recyclable nature of packaging?"

The application retrieves relevant information from the uploaded documents and uses the language model to generate a concise answer with a reference to the original PDF page.

## Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Natural Language Processing (NLP)
- Semantic Text Chunking
- Vector Embeddings
- Similarity Search
- Large Language Models (LLMs)
- LangChain Integration
- Local AI Model Deployment
- Flask Backend Development

## Future Improvements

- Support for multiple document formats.
- Improved retrieval using hybrid search.
- Conversation memory for multi-turn interactions.
- Document upload functionality through the web interface.
- Improved answer evaluation and retrieval accuracy.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning & AI Enthusiast | Python Developer

GitHub: https://github.com/Vedline2547
