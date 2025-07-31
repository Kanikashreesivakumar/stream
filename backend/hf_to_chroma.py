from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from datasets import load_dataset
import os

try:
    import boto3
    from smart_open import open as sopen
except ImportError:
    boto3 = None
    sopen = None

load_dotenv()

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")

def get_documents_from_hf(dataset_name="bigcode/the-stack-v2", language=None, max_docs=100):
    """Stream documents from HuggingFace dataset and return as LangChain Documents."""
    if language:
        ds = load_dataset(dataset_name, language, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    docs = []
    for i, sample in enumerate(ds):
        
        if "blob_id" in sample and "src_encoding" in sample and boto3 and sopen:
            session = boto3.Session(
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
            s3 = session.client("s3")
            s3_url = f"s3://softwareheritage/content/{sample['blob_id']}"
            with sopen(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
                content = fin.read().decode(sample["src_encoding"])
        else:
            
            content = sample.get("content", "")
        if content:
            docs.append(Document(page_content=content, metadata={"source": sample.get("path", "hf_sample")}))
        if len(docs) >= max_docs:
            break
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
    print("Downloading and preparing documents from HuggingFace dataset...")
    docs = get_documents_from_hf(max_docs=50)  # Change max_docs as needed
    print(f"Fetched {len(docs)} documents. Indexing...")
    vectorstore.add_documents(docs)
    vectorstore.persist()
    print(f"Indexed {len(docs)} documents from HuggingFace dataset.")

if __name__ == "__main__":
    main()