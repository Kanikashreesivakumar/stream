import streamlit as st
from backend.indexer import build_index_from_uploaded, get_retriever_from_vectordb
from backend.qa_chain import answer_query
import os


gemini_api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Classic RAG-Flow: User query-langchain-AI Generation Code Explainer", layout="wide")
st.title("🤖 DecodeX AI - Embed.Retrive.Explain")

with st.sidebar:
    st.header("Upload Code Files or ZIP")
    uploaded_files = st.file_uploader(
        "Upload .py, .js, or .zip files",
        type=["py", "js", "zip"],
        accept_multiple_files=True
    )
    if st.button("Build/Reload Index"):
        if uploaded_files:
            with st.spinner("Indexing uploaded files, please wait..."):
                vectordb = build_index_from_uploaded(uploaded_files)
                st.session_state['vectordb'] = vectordb
                st.session_state['index_built'] = True
            st.success("Index built and loaded!")
        else:
            st.warning("Please upload code files or a ZIP.")

if 'index_built' not in st.session_state:
    st.session_state['index_built'] = False

st.markdown("---")
st.subheader("💬 Chat with your codebase")

if st.session_state.get('index_built', False):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.chat_input("Ask a question about your codebase...")
    if user_input:
        with st.spinner("Thinking..."):
            retriever = get_retriever_from_vectordb(st.session_state['vectordb'])
            response = answer_query(user_input, retriever=retriever)
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("bot", response))

    for sender, msg in st.session_state["chat_history"]:
        if sender == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)
else:
    st.info("Please upload files and build the index using the sidebar.")
