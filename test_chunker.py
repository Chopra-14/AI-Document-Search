from src.pdf_loader import load_pdf
from src.chunker import create_chunks


pages = load_pdf("data/documents/sample.pdf")

chunks = create_chunks(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print("\nCHUNK:", i + 1)
    print("PAGE:", chunk["page"])
    print(chunk["text"])
    print("----------------")