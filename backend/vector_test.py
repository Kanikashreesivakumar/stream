from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
import os

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = Chroma(collection_name="my_docs", embedding_function=embeddings)


docs = [Document(page_content="Your text here", metadata={"source": "manual"})]
vectorstore.add_documents(docs)

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

result = qa_chain({"query": "What is in my docs?"})
print(result["result"])