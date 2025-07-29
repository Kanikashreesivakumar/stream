# AI-Driven Code Explainer Chatbot

## Overview
This project is an AI-powered chatbot that can ingest code repositories (Python, JavaScript, etc.), index and retrieve code, and answer questions about code logic, flow, and suggest improvements. It uses LangChain, OpenAI or SentenceTransformers, and ChromaDB/FAISS for vector search, with a Streamlit UI frontend.

## Features
- Ingests and indexes code repositories
- Splits and embeds code for retrieval
- Answers questions about code logic and functions
- Explains code flow in natural language
- Suggests optimizations or bug fixes
- Simple Streamlit UI for interaction

## Setup Instructions

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   - Add your OpenAI API key to `backend/config.yaml`:
     ```yaml
     OPENAI_API_KEY: "your-openai-api-key-here"
     ```

3. **Run the App:**
   ```bash
   streamlit run app.py
   ```

4. **Usage:**
   - Enter the path to your code directory in the UI.
   - Click "Build/Reload Index" to index your codebase.
   - Ask questions about your codebase in the chat box.

## Secure API Key Setup

**Do NOT commit your OpenAI API key to the repository.**

Instead, set your API key as an environment variable before running the app:

**Windows (PowerShell):**
```
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Linux/Mac:**
```
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Then run:
```
streamlit run app.py
```

Your app will automatically read the API key from the environment.  
This keeps your secret safe and ensures you can push your code to GitHub without exposing sensitive information.

## Project Structure
```
├── app.py                # Streamlit UI entry point
├── backend/
│   ├── code_loader.py    # Loads and splits code files
│   ├── indexer.py        # Handles embeddings and vector DB
│   ├── qa_chain.py       # LangChain pipeline for Q&A
│   └── config.yaml       # Configurations (API keys, paths, etc.)
├── requirements.txt
├── README.md
└── data/
    └── (indexed DB, code files, etc.)
```

## Example Questions
- "Explain the function `foo` in my codebase."
- "What does the main loop do?"
- "Suggest improvements for the data loader."
- "Are there any bugs in the authentication logic?"

## Notes
- For offline embeddings, SentenceTransformers is used by default. To use OpenAI embeddings, modify `indexer.py` accordingly.
- For large codebases, ensure you have sufficient RAM and disk space.

---
