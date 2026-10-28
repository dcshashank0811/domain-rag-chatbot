from io import BytesIO
from pypdf import PdfReader


def extract_pdf_pages(pdf_bytes: bytes, document_name: str) -> list[dict]:
    """Extract non-empty page text and preserve document/page metadata."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "text": text,
                "document": document_name,
                "page": page_number,
            })
    return pages
