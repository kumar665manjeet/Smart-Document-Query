import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFaceHub

# Ensure the API token is set in the environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_API_TOKEN"

def load_chunk_persist_pdf():
    """
    Load PDF documents, split them into chunks, and persist them in a ChromaDB vector store.

    Returns:
        vectordb (Chroma): The ChromaDB vector store containing the chunked PDF documents.
    """
    pdf_folder_path = "/Your/Dataset Path/"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if not client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="/path/langchain_tests/store/"
    )
    vectordb.persist()
    return vectordb

def create_agent_chain():
    """
    Create a QA chain using a HuggingFace model.

    Returns:
        chain (Chain): The QA chain.
    """
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature": 0.1, "max_length": 700})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    """
    Get a response from the LLM based on the query and the content of the PDF documents.

    Args:
        query (str): The query to ask the LLM.

    Returns:
        answer (str): The response from the LLM.
    """
    vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    if "Helpful Answer:" in answer:
        answer = answer.split("Helpful Answer:")[-1].strip()
    return answer

# Streamlit UI
# ===============
st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("Query PDF Source")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    response = get_llm_response(form_input)
    st.write(response)
