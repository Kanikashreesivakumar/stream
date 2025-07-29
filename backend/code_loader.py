import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_code_files(code_dir, file_types=(".py", ".js")):
    """Load code files from a directory using LangChain's DirectoryLoader."""
    loader = DirectoryLoader(
        code_dir,
        glob="**/*",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()
    # Filter by file extension
    docs = [doc for doc in docs if os.path.splitext(doc.metadata['source'])[1] in file_types]
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
