import os
import streamlit as st
from src.embeddings import create_embeddings
from src.vector_store import collection
import ollama

try:
    from groq import Groq
except ImportError:
    Groq = None


def ask_question_stream(question, selected_documents=None, chat_history=None, groq_api_key=None):
    """
    RAG with streaming response. Supports both local Ollama and Cloud Groq API.
    """
    # Check for Groq API key from parameter, secrets, or environment
    if not groq_api_key:
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY", None)
        except Exception:
            groq_api_key = None
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", None)

    # Handle general conversational queries / greetings cleanly
    q_clean = question.strip().lower()
    greetings = ["hi", "hello", "hey", "how can you help me", "how can u help me",
                 "what can you do", "who are you"]
    if any(q_clean == g for g in greetings) or (len(q_clean) <= 10 and any(g in q_clean for g in ["hi", "hello", "hey", "help"])):
        def greeting_generator():
            greeting_text = (
                "👋 Hello! I am your **AI Document Search Assistant**.\n\n"
                "Here is how I can help you:\n"
                "- 📄 **Summarize Documents:** Get quick overviews of your uploaded PDFs.\n"
                "- ❓ **Answer Questions:** Ask specific questions about your study material, notes, or docs.\n"
                "- 🎯 **Generate Exam Questions:** Ask for 2-mark, 5-mark, or 7-mark questions from your docs.\n"
                "- 📑 **Provide Citations:** Every answer references exact document pages."
            )
            yield greeting_text
        return greeting_generator(), []

    # Check if collection is empty
    if collection.count() == 0:
        def empty_generator():
            yield "No documents found in the database. Please upload a PDF in the sidebar first."
        return empty_generator(), []

    where_filter = None
    if selected_documents:
        possible_names = []
        for d in selected_documents:
            possible_names.extend([d, f"{d}.pdf", d.lower(), f"{d.lower()}.pdf"])
        possible_names = list(set(possible_names))
        if len(possible_names) == 1:
            where_filter = {"document": possible_names[0]}
        else:
            where_filter = {"document": {"$in": possible_names}}

    # Search query with recent conversational context for follow-ups
    search_query = question
    if chat_history:
        recent_user_msgs = [m["content"] for m in chat_history if m["role"] == "user"][-2:]
        if recent_user_msgs:
            search_query = recent_user_msgs[-1] + " " + question

    # Create embedding for search
    question_embedding = create_embeddings([search_query])[0]

    # Retrieve top 4 most relevant chunks
    n_fetch = min(4, collection.count())
    results = None

    if where_filter:
        try:
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=n_fetch,
                where=where_filter
            )
        except Exception:
            results = None

    # Fallback to global search if filter returned nothing
    if not results or not results.get("documents") or not results["documents"][0]:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_fetch
        )

    documents = results["documents"][0] if (results and results["documents"]) else []
    metadatas = results["metadatas"][0] if (results and results["metadatas"]) else []

    if not documents:
        def no_doc_generator():
            yield "The uploaded documents do not contain relevant information to answer this question."
        return no_doc_generator(), []

    # Format context chunks
    formatted_contexts = []
    sources = []

    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        doc_name = meta.get("document", "Document")
        page_num = meta.get("page", "N/A")

        formatted_contexts.append(
            f"--- Context Chunk {idx} [Doc: {doc_name} | Page: {page_num}] ---\n{doc}"
        )

        sources.append({
            "page": page_num,
            "document": doc_name,
            "snippet": doc
        })

    context_str = "\n\n".join(formatted_contexts)

    # Conversation history snippet
    history_str = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        if history_lines:
            history_str = "\n".join(history_lines)

    prompt = f"""You are an expert AI document assistant and tutor.
Answer the user's question clearly, thoroughly, and accurately based ONLY on the provided document context.

Instructions:
- Use conversation history to understand follow-up questions (e.g. "expand", "for 7 marks").
- For 2-mark questions: 1-2 sentence concise answer.
- For 5-mark / 7-mark / detailed questions: provide structured points with bold headings.
- Use clear bullet points and clean formatting.
- Rely STRICTLY on the document context provided below.

--- Conversation History ---
{history_str if history_str else "None"}

--- Document Context ---
{context_str}

--- User Question ---
{question}

--- Detailed Answer ---"""

    def stream_generator():
        # 1. Try Groq Cloud API if key is provided
        if groq_api_key and Groq:
            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a precise AI document assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=800,
                    stream=True
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                return
            except Exception as e:
                yield f"⚠️ Groq API Error: {str(e)}\n\nFalling back to local Ollama..."

        # 2. Try Local Ollama
        try:
            stream = ollama.generate(
                model="llama3.2",
                prompt=prompt,
                stream=True,
                options={
                    "temperature": 0.2,
                    "num_ctx": 2048,
                    "num_predict": 512
                }
            )
            for chunk in stream:
                yield chunk["response"]
        except Exception as e:
            # Helpful error guide for Cloud Deployment
            yield (
                f"⚠️ **Could not connect to local Ollama:** `{str(e)}`\n\n"
                "**Deploying on Streamlit Cloud?**\n"
                "Streamlit Cloud does not have local Ollama installed. To enable instant AI answers on the cloud:\n"
                "1. Get a **Free Groq API Key** in 30 seconds from [console.groq.com](https://console.groq.com/keys).\n"
                "2. Enter your Groq API Key in the **Sidebar Settings** or in Streamlit Cloud Secrets (`GROQ_API_KEY`)."
            )

    return stream_generator(), sources


def ask_question(question, selected_documents=None, chat_history=None, groq_api_key=None):
    """
    Synchronous fallback for tests and legacy callers.
    """
    stream, sources = ask_question_stream(question, selected_documents, chat_history, groq_api_key)
    answer = "".join(list(stream))
    return answer, sources