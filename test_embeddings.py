from src.embeddings import create_embeddings


texts = [
    "What is machine learning?",
    "Machine learning allows computers to learn from data.",
    "The capital of France is Paris."
]

embeddings = create_embeddings(texts)

print("Number of texts:", len(texts))
print("Number of embeddings:", len(embeddings))
print("Embedding size:", len(embeddings[0]))
print("First embedding:")
print(embeddings[0])