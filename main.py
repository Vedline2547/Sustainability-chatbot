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

# FUNCTIONS

def load_all_documents(pdf_folder):
    """
    Loads all PDF documents from the specified folder using PDFPlumberLoader.
    Returns a list of document objects.
    """
    all_docs = []

    if not os.path.exists(pdf_folder):
        print(f"PDF folder '{pdf_folder}' not found. Make sure your PDFs are in the folder.")
        return all_docs

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Loading: {pdf_path}")

            loader = PDFPlumberLoader(pdf_path)
            all_docs.extend(loader.load())

    return all_docs


def get_vector_store(documents, embedder, index_dir):
    """
    Creates or loads a FAISS vector store using the specified embedder.
    """

    if os.path.exists(index_dir):
        vector = FAISS.load_local(
            index_dir,
            embedder,
            allow_dangerous_deserialization=True
        )
        print("Loaded vector store from disk.")

    else:
        # Split documents into semantic chunks
        text_splitter = SemanticChunker(embedder)
        docs_chunks = text_splitter.split_documents(documents)

        vector = FAISS.from_documents(docs_chunks, embedder)
        vector.save_local(index_dir)

        print("Created and saved new vector store.")

    return vector


def build_llm_chain(llm):
    """
    Creates an LLMChain using the provided LLM and QA prompt.
    """

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

    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        verbose=True
    )

    return llm_chain


def build_combined_documents_chain(llm_chain):
    """
    Creates a StuffDocumentsChain for combining retrieved documents
    using a custom document prompt.
    """

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\nContent: {page_content}\nSource: {source}"
    )

    combined_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt
    )

    return combined_chain


def build_qa_chain(retriever, combined_documents_chain):
    """
    Builds and returns a RetrievalQA chain.
    """

    qa_chain = RetrievalQA(
        combine_documents_chain=combined_documents_chain,
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return qa_chain


def initialize_chain():
    """
    Initializes the full RetrievalQA chain by:
    1. Loading PDF documents from 'static/uploads'.
    2. Creating or loading a FAISS vector store.
    3. Building the LLM and associated chains.
    """

    pdf_folder = os.path.join("static", "uploads")
    all_docs = load_all_documents(pdf_folder)

    if not all_docs:
        print("No PDF documents loaded.")
        return None

    index_dir = "faiss_index"
    embedder = HuggingFaceEmbeddings()

    vector = get_vector_store(all_docs, embedder, index_dir)

    retriever = vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Define the LLM
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        num_ctx=2048,
        num_predict=256,
        keep_alive="30m"
    )

    llm_chain = build_llm_chain(llm)

    combined_documents_chain = build_combined_documents_chain(llm_chain)

    qa_chain = build_qa_chain(
        retriever,
        combined_documents_chain
    )

    return qa_chain


def get_response(qa_chain, question):
    """
    Runs the QA chain with the provided question.
    Returns the answer and a relative URL for the source PDF.
    """

    if qa_chain is None:
        return "QA chain is not initialized.", None

    response = qa_chain.invoke({"query": question})

    answer_text = response.get("result", "I don't know")

    pdf_url = None

    source_documents = response.get("source_documents", [])

    if source_documents:
        doc = source_documents[0]
        metadata = doc.metadata

        page_num = metadata.get("page", 0)
        source_doc = metadata.get("source", "")

        # Normalize path separators
        normalized_source = source_doc.replace("\\", "/")

        # Extract the path relative to the static folder
        if normalized_source.lower().startswith("static/"):
            normalized_source = normalized_source[len("static/"):]

        pdf_url = f"/static/{normalized_source}#page={page_num + 1}"

    return answer_text, pdf_url


# INITIALIZE QA CHAIN

qa_chain = initialize_chain()

sample_question = (
    "ESSENTIAL REQUIREMENTS ON THE COMPOSITION AND THE REUSABLE "
    "AND RECOVERABLE, INCLUDING RECYCLABLE, NATURE OF PACKAGING"
)

answer, pdf_url = get_response(qa_chain, sample_question)

print("Question:", sample_question)
print("Answer:", answer)
print("PDF URL:", pdf_url)