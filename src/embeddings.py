import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts):
    if not texts:
        return []
    model = get_embedding_model()
    return model.encode(texts)