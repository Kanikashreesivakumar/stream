from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.indexer import get_retriever_from_vectordb
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    return {}

def answer_query(query, retriever=None):
    
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment variables.")

    # Create Gemini LLM wrapper for LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",  # or another supported model
        google_api_key=gemini_api_key,
        temperature=0.2,
        max_output_tokens=2048,
    )

    if retriever is not None:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )
        result = qa_chain({"query": query})
        return result["result"]
    else:
   
        return llm.invoke(query)
