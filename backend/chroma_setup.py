from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import os

from datasets import load_dataset

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")

def load_docs():
    docs = []
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": fname}))
    return docs

def load_docs_from_hf():
    dataset = load_dataset("squad", split="train[:100]")
    print("Number of documents in dataset:", len(dataset))
    docs = []
    for item in dataset:
        context = item.get("context", "")
        question = item.get("question", "")
        docs.append(Document(page_content=f"Q: {question}\nA: {context}", metadata={"id": item.get("id", "")}))
    print(f"Loaded {len(docs)} documents from Hugging Face dataset.")
    return docs

def main():
    # Use a multilingual embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    # You can switch between local or HF docs as needed
    docs = load_docs_from_hf()  # or docs = load_docs()
    vectorstore.add_documents(docs)
    vectorstore.persist()
    print(f"Indexed {len(docs)} documents.")

def check_chroma_docs():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    num_docs = vectorstore._collection.count()
    print(f"Chroma vectorstore contains {num_docs} documents.")

if __name__ == "__main__":
    main()
    check_chroma_docs()