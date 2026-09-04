import chromadb


# Create ChromaDB client
client = chromadb.PersistentClient(
    path="chroma_db"
)


# Create a fresh collection
collection = client.get_or_create_collection(
    name="documents_v2"
)

def add_documents(chunks, embeddings, document_name):

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        document_id = f"{document_name}_{index}"

        ids.append(document_id)

        documents.append(chunk["text"])

        metadatas.append({
            "page": chunk["page"],
            "document": document_name
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def delete_document(document_name):
    collection.delete(
        where={
            "document": document_name
        }
    )