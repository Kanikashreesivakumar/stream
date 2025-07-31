from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")

def answer_query(query):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    result = qa_chain({"query": query})
    return result["result"]

if __name__ == "__main__":
    q = input("Ask a question: ")
    print(answer_query(q))