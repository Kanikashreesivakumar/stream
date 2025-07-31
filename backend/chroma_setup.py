from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import os

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

def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    docs = load_docs()
    vectorstore.add_documents(docs)
    vectorstore.persist()
    print(f"Indexed {len(docs)} documents.")

def check_chroma_docs():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    # Load persisted data
    num_docs = vectorstore._collection.count()
    print(f"Chroma vectorstore contains {num_docs} documents.")

if __name__ == "__main__":
    main()               # This will index and persist your documents
    check_chroma_docs()  # Now this will show the correct count