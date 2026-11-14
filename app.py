from flask import Flask, render_template, request, url_for
import os
import json

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains import RetrievalQA


# Initialize the Flask app
app = Flask(__name__)


# Load PDF documents
all_docs = []
pdf_folder = os.path.join("static", "uploads")

if os.path.exists(pdf_folder):
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Loading: {pdf_path}")

            loader = PDFPlumberLoader(pdf_path)
            all_docs.extend(loader.load())
else:
    print(f"PDF folder '{pdf_folder}' not found.")


# Create or load the vector store
index_dir = "faiss_index"
embedder = HuggingFaceEmbeddings()

if all_docs:

    if os.path.exists(index_dir):
        # Load existing vector store
        vector = FAISS.load_local(
            index_dir,
            embedder,
            allow_dangerous_deserialization=True
        )

        print("Loaded vector store from disk.")

    else:
        # Split documents into semantic chunks
        text_splitter = SemanticChunker(embedder)
        docs_chunks = text_splitter.split_documents(all_docs)

        vector = FAISS.from_documents(docs_chunks, embedder)
        vector.save_local(index_dir)

        print("Created and saved new vector store.")

    retriever = vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

else:
    retriever = None
    print("No PDF documents loaded.")


# Initialize Ollama model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    keep_alive="30m"
)


# Define prompt for QA
prompt = """
1. Use the following pieces of context to answer the question.
2. If you don't know the answer, simply say "I don't know".
   Do not make up information.
3. Keep the answer crisp and limited to 3-4 sentences.

Context:
{context}

Question:
{question}

Helpful answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)


# Build LLM Chain
llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    verbose=True
)


# Define document prompt for combining retrieved documents
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\nContent: {page_content}\nSource: {source}"
)


# Create StuffDocumentsChain
combined_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt
)


# Create RetrievalQA chain
if retriever is not None:
    qa_chain = RetrievalQA(
        combine_documents_chain=combined_documents_chain,
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
else:
    qa_chain = None


# Get response from the QA chain
def get_response(question):

    if qa_chain is None:
        return "No PDF documents are available.", None

    response = qa_chain.invoke({"query": question})

    answer_text = response.get("result", "I don't know")

    print("The response from the chain is:", response)

    pdf_url = None

    source_documents = response.get("source_documents", [])

    if source_documents:
        doc = source_documents[0]
        metadata = doc.metadata

        source_doc = metadata.get("source", "")
        page_num = metadata.get("page", 0)

        # Normalize path separators
        normalized_source = source_doc.replace("\\", "/")

        if normalized_source.lower().startswith("static/"):
            normalized_source = normalized_source[len("static/"):]

        pdf_url = f"/static/{normalized_source}#page={page_num + 1}"

        print("Returned URL is:", pdf_url)

    return answer_text, pdf_url


# Flask routes

# Landing page route
@app.route("/")
def landing():
    return render_template("landing.html")


# Chat page route
@app.route("/chat", methods=["GET", "POST"])
def chat():

    if request.method == "POST":

        question = request.form.get("question", "").strip()

        if not question:
            return render_template(
                "chat.html",
                question=question,
                answer="Please enter a question.",
                pdf_url=None
            )

        answer, pdf_url = get_response(question)

        return render_template(
            "chat.html",
            question=question,
            answer=answer,
            pdf_url=pdf_url
        )

    return render_template(
        "chat.html",
        question="",
        answer="",
        pdf_url=""
    )


if __name__ == "__main__":
    app.run(debug=False)