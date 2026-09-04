from src.rag import ask_question


question = "What is quantum mechanics?"

answer, sources = ask_question(question)

print("\nANSWER:")
print(answer)

print("\nSOURCE PAGES:")

for source in sources:
    print("Page:", source["page"])