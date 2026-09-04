from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import add_documents


pages = load_pdf("data/documents/sample.pdf")

chunks = create_chunks(pages)

texts = [chunk["text"] for chunk in chunks]

embeddings = create_embeddings(texts)

add_documents(chunks, embeddings)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))
print("Embeddings created:", len(embeddings))
print("Documents stored in ChromaDB successfully!")