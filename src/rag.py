import os
import json
import urllib.request
import streamlit as st
from src.embeddings import create_embeddings
from src.vector_store import collection
import ollama

try:
    from groq import Groq
except ImportError:
    Groq = None


def ask_question_stream(question, selected_documents=None, chat_history=None, api_key=None, provider="auto"):
    """
    RAG with streaming response.
    Supports:
    1. Groq Cloud API (Llama 3.3 70B / Llama 3.1 8B)
    2. Google Gemini Cloud API (Gemini 1.5/2.0 Flash)
    3. Local Ollama (Llama 3.2)
    """
    # 1. Resolve API key if not explicitly passed
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", None) or st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            api_key = None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", None) or os.environ.get("GEMINI_API_KEY", None)

    if api_key:
        api_key = str(api_key).strip().strip("'\"")

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

    # Conversation history snippet (compact to save context tokens)
    history_str = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-2:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content'][:250]}")
        if history_lines:
            history_str = "\n".join(history_lines)

    prompt = f"""You are an expert AI document assistant and tutor.
Answer the user's question clearly, thoroughly, and accurately based ONLY on the provided document context.

Instructions:
- Use conversation history to understand follow-up questions.
- For 2-mark questions: 1-2 sentence concise answer.
- For 5-mark / 7-mark / detailed questions: provide structured points with bold headings.
- Rely STRICTLY on the document context provided below.

--- Conversation History ---
{history_str if history_str else "None"}

--- Document Context ---
{context_str[:3000]}

--- User Question ---
{question}

--- Detailed Answer ---"""

    def stream_generator():
        # Case 1: Google Gemini API (Key starts with AIzaSy...)
        if api_key and (api_key.startswith("AIzaSy") or provider == "gemini"):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800}
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
                    yield text
                return
            except Exception as e:
                yield f"⚠️ **Google Gemini API Error:** `{str(e)}`\n\n"

        # Case 2: Groq Cloud API (Key starts with gsk_...)
        if api_key and Groq:
            try:
                client = Groq(api_key=api_key)
                
                # Fetch live models list directly from Groq for this account
                available_models = []
                try:
                    for m in client.models.list().data:
                        m_id = getattr(m, 'id', '')
                        # Exclude low-rate-limit preview models, audio, and safety filters
                        if m_id and not any(x in m_id.lower() for x in ['whisper', 'guard', 'embed', 'tts', 'moderation', 'qwen']):
                            available_models.append(m_id)
                except Exception:
                    available_models = []

                # Filter strictly for text LLMs (Llama, Gemma, DeepSeek)
                chat_models = []
                for m in available_models:
                    m_lower = m.lower()
                    if any(k in m_lower for k in ['llama', 'gemma', 'deepseek']) and not any(x in m_lower for x in ['canopy', 'orpheus', 'playht', 'audio', 'tts', 'guard']):
                        chat_models.append(m)

                # Match best model by substring preference
                chosen_model = None
                for pref in ["llama-3.3", "llama-3.1", "70b", "8b", "llama", "gemma"]:
                    matches = [m for m in chat_models if pref in m.lower()]
                    if matches:
                        chosen_model = matches[0]
                        break

                if not chosen_model:
                    chosen_model = chat_models[0] if chat_models else "llama-3.1-70b-versatile"

                stream = client.chat.completions.create(
                    model=chosen_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=400,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_text = chunk.choices[0].delta.content
                        if delta_text:
                            yield delta_text
                return
            except Exception as e:
                err_str = str(e)
                if "401" in err_str or "invalid_api_key" in err_str:
                    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else f"`{api_key}` (Too short)"
                    yield (
                        f"🔑 **Groq API Authentication Error (401)**\n\n"
                        f"- **Key Received:** `{masked_key}` (Length: {len(api_key)} chars)\n\n"
                        "The key was rejected by Groq. Please make sure:\n"
                        "1. Go to **[console.groq.com/keys](https://console.groq.com/keys)**\n"
                        "2. Click **Create API Key** and copy the **entire key** immediately.\n"
                        "3. Paste it directly into the **`🔑 Cloud API Key`** box in the sidebar without spaces."
                    )
                else:
                    yield f"⚠️ **Groq API Error:** `{err_str}`"
                return

        # Case 3: Local Ollama (when running on localhost without cloud key)
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
            yield (
                f"⚠️ **Could not connect to local Ollama:** `{str(e)}`\n\n"
                "**Running on Streamlit Cloud?**\n"
                "Streamlit Cloud does not have local Ollama installed. To enable instant AI answers on the cloud:\n\n"
                "1. Get a **Free Cloud API Key** from **[console.groq.com/keys](https://console.groq.com/keys)** (starts with `gsk_...`) or Google AI Studio (starts with `AIzaSy...`).\n"
                "2. Paste it in the **Sidebar Settings**!"
            )

    return stream_generator(), sources


def ask_question(question, selected_documents=None, chat_history=None, api_key=None):
    """
    Synchronous fallback for tests and legacy callers.
    """
    stream, sources = ask_question_stream(question, selected_documents, chat_history, api_key)
    answer = "".join(list(stream))
    return answer, sources