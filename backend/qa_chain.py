from langchain.chains import RetrievalQA
from langchain_community.llms import Anthropic
from backend.indexer import get_retriever_from_vectordb
import os
import yaml

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    return {}

def answer_query(query, retriever=None):
    
    llm = Anthropic(model="claude-3-haiku-20240307", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
