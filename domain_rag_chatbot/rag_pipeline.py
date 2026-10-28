import os
from document_loader import extract_pdf_pages
from vector_store import create_vector_store
from prompt import SYSTEM_PROMPT


def build_index(pdfs: list[dict]):
    pages = []
    for pdf in pdfs:
        if not pdf["name"].lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file: {pdf['name']}")
        pages.extend(extract_pdf_pages(pdf["bytes"], pdf["name"]))
    if not pages:
        raise ValueError("No extractable text was found. Scanned PDFs need an OCR extension.")
    return create_vector_store(pages)


def answer_question(vector_store, question: str, top_k: int = 4) -> dict:
    """Retrieve relevant chunks and ask a configured LLM for a grounded answer."""
    matches = vector_store.similarity_search_with_score(question, k=top_k)
    if not matches:
        return {
            "answer": "I could not find this information in the uploaded documents.",
            "sources": [],
        }

    context_parts = []
    sources = []
    for doc, distance in matches:
        meta = doc.metadata
        context_parts.append(
            f"[Source: {meta.get('document', 'Unknown')}, page {meta.get('page', '?')}]\n"
            f"{doc.page_content}"
        )
        # FAISS L2 distance is not a calibrated probability; display it as a distance.
        sources.append({
            "document": meta.get("document", "Unknown"),
            "page": meta.get("page", "?"),
            "score": float(distance),
            "text": doc.page_content,
        })

    context = "\n\n---\n\n".join(context_parts)
    answer = generate_answer(question, context)
    return {"answer": answer, "sources": sources}


def generate_answer(question: str, context: str) -> str:
    """Use Groq if GROQ_API_KEY is set; otherwise provide a safe context-only fallback."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "LLM answer generation is not configured. Add your GROQ_API_KEY to the .env file "
            "and restart the app. The retrieved source passages are shown below so you can "
            "inspect the relevant document content."
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer only from the context.",
                },
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        return (
            "I couldn't generate an LLM answer. Check your API key, model setting, and network "
            f"connection. Error: {exc}"
        )
