from dotenv import load_dotenv

load_dotenv()  # must run before rag_pipeline reads GROQ_API_KEY

import streamlit as st
from rag_pipeline import build_index, answer_question

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Domain-Specific RAG Chatbot")
st.caption("Ask questions about your PDFs. Answers are grounded in retrieved document passages.")

with st.sidebar:
    st.header("1. Upload documents")
    uploads = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="For privacy, avoid uploading confidential documents unless you have permission.",
    )
    top_k = st.slider("Retrieved passages", min_value=3, max_value=5, value=4)
    process = st.button("Process documents", type="primary", use_container_width=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if process:
    if not uploads:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Extracting text, creating embeddings, and building the search index..."):
            try:
                pdfs = [{"name": f.name, "bytes": f.getvalue()} for f in uploads]
                st.session_state.rag_index = build_index(pdfs)
                st.session_state.messages = []
                st.success(f"Processed {len(uploads)} PDF file(s). You can now ask questions.")
            except Exception as exc:
                st.error(f"Could not process the documents: {exc}")

if st.session_state.rag_index is None:
    st.info("Upload PDF files in the sidebar and click **Process documents** to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['document']}**, page {source['page']} "
                            f"(distance: {source['score']:.3f}, lower = closer)"
                        )
                        st.caption(source["text"])

    question = st.chat_input("Ask a question about your uploaded PDFs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    result = answer_question(st.session_state.rag_index, question, top_k=top_k)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources", expanded=True):
                            for source in result["sources"]:
                                st.markdown(
                                    f"**{source['document']}**, page {source['page']} "
                                    f"(distance: {source['score']:.3f}, lower = closer)"
                                )
                                st.caption(source["text"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as exc:
                    st.error(f"An error occurred while answering: {exc}")

st.divider()
st.caption("Please verify answers before relying on them for legal, medical, financial, or other high-stakes decisions.")
