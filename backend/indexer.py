import os
import tempfile
import zipfile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_code_files_from_uploaded(files, file_types=(".py", ".js")):
    """Load code files from uploaded files (supports zip extraction)."""
    docs = []
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in files:
        filename = uploaded_file.name
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if filename.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
  
    for root, _, files in os.walk(temp_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1]
            if ext in file_types:
                fpath = os.path.join(root, fname)
                try:
                    loader = TextLoader(fpath)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Skipping {fpath} due to error: {e}")
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_in_memory_vectorstore(docs):
    embeddings = get_embeddings_model()
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=None 
    )
    return vectordb

def build_index_from_uploaded(files):
    docs = load_code_files_from_uploaded(files)
    chunks = split_documents(docs)
    vectordb = create_in_memory_vectorstore(chunks)
    return vectordb

def get_retriever_from_vectordb(vectordb):
    return vectordb.as_retriever()
