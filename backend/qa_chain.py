from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
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
    config = load_config()
    openai_api_key = config.get('OPENAI_API_KEY', None)
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
