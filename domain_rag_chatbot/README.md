# Domain-Specific RAG Chatbot for PDF Question Answering

A Streamlit app that accepts PDF documents, extracts page text, chunks it, embeds it with Sentence Transformers, stores vectors in FAISS, retrieves relevant passages, and generates grounded answers with Groq.

## Features
- Upload multiple PDFs
- Preserve source filename and page metadata
- Chunk text with overlap
- Sentence Transformers (`all-MiniLM-L6-v2`) embeddings
- FAISS similarity search
- Groq LLM answer generation with a context-only prompt
- Source passages and page numbers displayed in the UI
- Chat history and clear-chat control
- Safe warning for high-stakes information

## Requirements
- Python 3.10–3.12 recommended for dependency compatibility
- Internet connection for package/model downloads and Groq generation
- A Groq API key for generated answers (retrieval works without one)

## Setup (Windows / VS Code)
1. Extract the ZIP.
2. Open the `domain_rag_chatbot` folder in VS Code.
3. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to `.env` and add your Groq API key. You can get a key from the Groq developer console. Never commit `.env` to GitHub.
6. Start the app:
   ```powershell
   streamlit run app.py
   ```

If `py -3.11` is not installed, install a supported Python version and select that interpreter in VS Code. First launch may take longer because the embedding model is downloaded.

## Usage
1. Upload one or more text-based PDFs in the sidebar.
2. Click **Process documents**.
3. Ask questions in the chat box.
4. Expand **Sources** to inspect the retrieved passages and their document/page references.

## Project structure
```text
domain_rag_chatbot/
├── app.py
├── document_loader.py
├── rag_pipeline.py
├── vector_store.py
├── prompt.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── documents/
├── tests/
│   ├── test_questions.csv
│   └── evaluate.py
└── vector_store/
```

## Evaluation
Put sample PDFs in `documents/`, fill in `tests/test_questions.csv` (expected source and page), then run:
```powershell
python tests/evaluate.py
```
Rows with an expected source check retrieval (right file and page in the top results). Rows marked `Not available` check that the model refuses; this needs a `GROQ_API_KEY`. Rows with no expected source are skipped.

## Limitations
- Scanned/image-only PDFs are not OCR processed in this starter version.
- If no Groq API key is configured, the app displays retrieved passages but does not generate a natural-language answer.
- Similarity distance is a retrieval metric, not a confidence percentage.
- Verify important answers against the original documents.

## Suggested extensions
- OCR for scanned PDFs
- Persist/reload FAISS indexes
- Document collection filters
- User login and access control
- Retrieval evaluation and feedback buttons
- Deploy on Streamlit Community Cloud or another supported host

## Security
Do not upload confidential documents without authorization. Keep API keys in `.env` or deployment secrets. Treat model output as fallible.
