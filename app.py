import os
import json
import time
from datetime import datetime
import streamlit as st

from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import add_documents, delete_document, collection
import importlib
import src.rag
importlib.reload(src.rag)
from src.rag import ask_question_stream, ask_question

# Page Configuration
st.set_page_config(
    page_title="AI Document Search | RAG Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Real-Life Chatbot UI (User on Right, Assistant on Left)
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1050px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        background-color: rgba(99, 102, 241, 0.15);
        color: #6366f1;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .badge-db {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .welcome-banner {
        padding: 24px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 24px;
        text-align: center;
    }
    
    /* User Chat Bubble: Right-aligned speech bubble */
    div[data-testid="stChatMessage"]:nth-child(even),
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.16), rgba(129, 140, 248, 0.10)) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 18px 18px 4px 18px !important;
        margin-left: auto !important;
        margin-right: 0px !important;
        margin-bottom: 14px !important;
        max-width: 78% !important;
        text-align: right !important;
    }
    
    /* Ensure content inside user bubble aligns cleanly */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stMarkdownContainer"] {
        text-align: left !important;
    }

    /* Assistant Chat Bubble: Left-aligned speech bubble */
    div[data-testid="stChatMessage"]:nth-child(odd),
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        flex-direction: row !important;
        background-color: rgba(243, 244, 246, 0.05) !important;
        border: 1px solid rgba(209, 213, 219, 0.25) !important;
        border-radius: 18px 18px 18px 4px !important;
        margin-right: auto !important;
        margin-left: 0px !important;
        margin-bottom: 16px !important;
        max-width: 88% !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions for Chat History Persistence
CHAT_DIR = os.path.join("data", "chat_history")
os.makedirs(CHAT_DIR, exist_ok=True)


def get_all_chat_sessions():
    sessions = []
    if os.path.exists(CHAT_DIR):
        for fname in os.listdir(CHAT_DIR):
            if fname.endswith(".json"):
                fpath = os.path.join(CHAT_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sessions.append(data)
                except Exception:
                    pass
    # Sort sessions by latest updated time
    sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
    return sessions


def save_current_chat():
    if not st.session_state.current_session_id or not st.session_state.messages:
        return
    
    # Generate title from first user prompt if default
    first_prompt = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "New Chat")
    title = first_prompt[:35] + "..." if len(first_prompt) > 35 else first_prompt

    session_data = {
        "id": st.session_state.current_session_id,
        "title": title,
        "updated_at": time.time(),
        "updated_str": datetime.now().strftime("%b %d, %H:%M"),
        "messages": st.session_state.messages
    }

    fpath = os.path.join(CHAT_DIR, f"{st.session_state.current_session_id}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)


def delete_chat_session(session_id):
    fpath = os.path.join(CHAT_DIR, f"{session_id}.json")
    if os.path.exists(fpath):
        os.remove(fpath)


def start_new_session():
    new_id = f"chat_{int(time.time())}"
    st.session_state.current_session_id = new_id
    st.session_state.messages = []


# Initialize Session State
if "current_session_id" not in st.session_state or not st.session_state.current_session_id:
    start_new_session()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Navigation
with st.sidebar:
    st.title("⚡ DocuMind RAG")
    st.caption("AI-Powered Knowledge Retrieval Engine")
    st.divider()

    # Chat History Section
    st.subheader("📜 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_session()
        st.rerun()

    saved_sessions = get_all_chat_sessions()

    if saved_sessions:
        for sess in saved_sessions:
            s_id = sess["id"]
            s_title = sess.get("title", "Saved Chat")
            s_date = sess.get("updated_str", "")
            
            is_active = (s_id == st.session_state.current_session_id)
            label = f"{'💬' if not is_active else '👉'} {s_title}"

            col_btn, col_del = st.columns([0.8, 0.2])
            with col_btn:
                if st.button(label, key=f"sess_{s_id}", use_container_width=True):
                    st.session_state.current_session_id = s_id
                    st.session_state.messages = sess.get("messages", [])
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"delsess_{s_id}", help="Delete chat history"):
                    delete_chat_session(s_id)
                    if st.session_state.current_session_id == s_id:
                        start_new_session()
                    st.toast("Chat deleted", icon="🗑️")
                    st.rerun()
    else:
        st.caption("No previous chats saved yet.")

    st.divider()

    # Upload Section
    st.subheader("📤 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF documents to index into ChromaDB."
    )

    if uploaded_files:
        document_folder = os.path.join("data", "documents")
        os.makedirs(document_folder, exist_ok=True)

        for uploaded_file in uploaded_files:
            document_name = os.path.splitext(uploaded_file.name)[0]
            pdf_path = os.path.join(document_folder, uploaded_file.name)

            if not os.path.exists(pdf_path):
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Indexing `{uploaded_file.name}`..."):
                    pages = load_pdf(pdf_path)
                    chunks = create_chunks(pages)
                    texts = [chunk["text"] for chunk in chunks]
                    embeddings = create_embeddings(texts)
                    add_documents(chunks, embeddings, document_name)

                st.toast(f"Indexed: {uploaded_file.name}", icon="✅")

    st.divider()

    # Active Documents Manager
    st.subheader("📚 Knowledge Base")
    document_folder = os.path.join("data", "documents")
    indexed_files = []
    if os.path.exists(document_folder):
        indexed_files = [f for f in os.listdir(document_folder) if f.lower().endswith(".pdf")]

    if indexed_files:
        doc_names = [os.path.splitext(f)[0] for f in indexed_files]
        selected_docs = st.sidebar.multiselect(
            "🎯 Search Scope Filter",
            options=doc_names,
            default=[],
            help="Select specific document(s) to search. Leave empty to search ALL documents."
        )

        for pdf_file in indexed_files:
            doc_name = os.path.splitext(pdf_file)[0]
            col_txt, col_del = st.columns([0.78, 0.22])
            with col_txt:
                st.markdown(f"<div style='font-size: 0.88rem; word-break: break-word;'>📄 <b>{pdf_file}</b></div>", unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"Delete {pdf_file}"):
                    delete_document(doc_name)
                    file_path = os.path.join(document_folder, pdf_file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    st.toast(f"Deleted {pdf_file}", icon="🗑️")
                    st.rerun()
    else:
        selected_docs = []
        st.info("No documents uploaded yet.")

    st.divider()

    # Stats & Actions
    st.subheader("📊 Stats")
    total_vectors = collection.count() if collection else 0
    st.markdown(f"- **Indexed Documents:** `{len(indexed_files)}`")
    st.markdown(f"- **Total Chunks:** `{total_vectors}`")
    st.markdown(f"- **LLM:** `Ollama / llama3.2`")

    st.divider()

    # Chat Export Download
    if st.session_state.messages:
        chat_md = f"# 📚 AI Document Search - Chat Export\n*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n"
        for m in st.session_state.messages:
            role_label = "### 👤 User" if m["role"] == "user" else "### 🤖 Assistant"
            chat_md += f"{role_label}\n\n{m['content']}\n\n"
            if m.get("sources"):
                chat_md += "#### 📑 Referenced Sources:\n"
                for s_idx, src in enumerate(m["sources"], 1):
                    chat_md += f"- **Source {s_idx}:** `{src.get('document', 'Doc')}` (Page {src.get('page', 'N/A')})\n"
                chat_md += "\n"
            chat_md += "---\n\n"

        st.download_button(
            label="📥 Download Chat (.md)",
            data=chat_md,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            help="Download the current chat notes with all answers and page citations in Markdown format."
        )

# Main Interface
st.title("💬 AI Document Search")
st.markdown("""
<span class="badge">Model: Llama 3.2</span>
<span class="badge badge-db">VectorDB: ChromaDB</span>
""", unsafe_allow_html=True)
st.write("")

# Welcome Banner when current chat is empty
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-banner">
        <h2>👋 Welcome to your AI Document Assistant</h2>
        <p style="color: #6b7280; margin-bottom: 12px;">Upload PDFs in the sidebar, ask questions, and your chat history will be automatically saved.</p>
    </div>
    """, unsafe_allow_html=True)

    if indexed_files:
        st.subheader("💡 Quick Starter Prompts")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📌 Summarize the key concepts in my document", use_container_width=True):
                st.session_state.pending_prompt = "Summarize the key concepts in my document."
                st.rerun()
        with col2:
            if st.button("🔍 What are the main conclusions or findings?", use_container_width=True):
                st.session_state.pending_prompt = "What are the main conclusions or findings?"
                st.rerun()

# Display Current Session Messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-top: 10px; margin-bottom: 16px;">
            <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); color: #ffffff; padding: 12px 18px; border-radius: 18px 18px 2px 18px; max-width: 78%; font-size: 0.96rem; line-height: 1.5; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📑 Referenced Sources & Citations"):
                    for idx, src in enumerate(message["sources"], start=1):
                        st.markdown(f"**Source {idx}:** `{src['document']}` (Page {src['page']})")
                        if "snippet" in src and src["snippet"]:
                            st.markdown(f"> {src['snippet'][:350]}...\n")

# Handle Quick Prompt Selection
prompt_input = st.chat_input("Ask a question about your uploaded documents...")

if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# Process User Input
if prompt_input:
    if not indexed_files:
        st.warning("⚠️ Please upload at least one PDF document in the sidebar before asking a question.")
    else:
        # Display user message immediately on RIGHT
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-top: 10px; margin-bottom: 16px;">
            <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); color: #ffffff; padding: 12px 18px; border-radius: 18px 18px 2px 18px; max-width: 78%; font-size: 0.96rem; line-height: 1.5; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);">
                {prompt_input}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        # Generate response on LEFT with Real-Time Streaming
        with st.chat_message("assistant", avatar="🤖"):
            stream_gen, sources = ask_question_stream(
                prompt_input,
                selected_documents=selected_docs,
                chat_history=st.session_state.messages
            )
            # Stream tokens live word-by-word like ChatGPT
            answer = st.write_stream(stream_gen)

            if sources:
                with st.expander("📑 Referenced Sources & Citations"):
                    for idx, src in enumerate(sources, start=1):
                        st.markdown(f"**Source {idx}:** `{src['document']}` (Page {src['page']})")
                        if "snippet" in src and src["snippet"]:
                            st.markdown(f"> {src['snippet'][:350]}...\n")

            # Store in session state and save chat file to disk
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            save_current_chat()