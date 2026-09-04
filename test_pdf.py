from src.pdf_loader import load_pdf


pages = load_pdf("data/documents/sample.pdf")

for page in pages:
    print("PAGE:", page["page"])
    print(page["text"][:500])
    print("----------------")