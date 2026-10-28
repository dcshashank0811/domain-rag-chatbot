from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(pages: list[dict]) -> FAISS:
    """Chunk page text, embed chunks, and create a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            if chunk.strip():
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "document": page["document"],
                        "page": page["page"],
                    },
                ))
    if not docs:
        raise ValueError("No readable text was found in the uploaded PDFs.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)
