#!/usr/bin/env python3
"""
Web-based File Q&A Interface - ChatGPT-style
Modern chat interface for document Q&A using Streamlit
"""

import os
import sys
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import tempfile
from typing import Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.file_qa import FileQA
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Lab Lens - File Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"  # Ensure sidebar is expanded by default
)

# Custom CSS for ChatGPT-like dark interface
st.markdown("""
<style>
    /* Hide Streamlit branding but keep sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Don't hide header - it contains the sidebar toggle button */
    /* header {visibility: hidden;} */
    
    /* Main container - full height layout */
    .main {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Block container - ensure proper spacing */
    .block-container {
        padding-bottom: 100px !important; /* Space for fixed input */
    }
    
    /* Fixed chat input at bottom - always visible */
    .chat-input-container {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background-color: #0e1117 !important;
        padding: 1rem !important;
        border-top: 1px solid #343541 !important;
        z-index: 999 !important;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Adjust for sidebar */
    [data-testid="stSidebar"][aria-expanded="true"] ~ .main .chat-input-container {
        left: 21rem !important;
    }
    
    @media (max-width: 768px) {
        [data-testid="stSidebar"] ~ .main .chat-input-container {
            left: 0 !important;
        }
    }
    
    /* Ensure page is scrollable */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        overflow-y: auto;
    }
    
    /* Messages area - allow natural scrolling */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Scroll to bottom button (appears when user scrolls up) */
    .scroll-to-bottom-btn {
        position: fixed;
        bottom: 100px;
        right: 2rem;
        background-color: #40414f;
        border: 1px solid #565869;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: none;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 998;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    
    .scroll-to-bottom-btn:hover {
        background-color: #4a4b59;
        transform: scale(1.1);
    }
    
    .scroll-to-bottom-btn.visible {
        display: flex;
    }
    
    /* Adjust for sidebar */
    [data-testid="stSidebar"][aria-expanded="true"] ~ .main .scroll-to-bottom-btn {
        right: calc(2rem + 21rem);
    }
    
    /* Sidebar styling - ChatGPT-like - FORCE VISIBILITY */
    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        padding: 1rem !important;
        min-width: 280px !important;
        max-width: 350px !important;
        visibility: visible !important;
        display: block !important;
        position: relative !important;
        z-index: 100 !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 280px !important;
        visibility: visible !important;
        display: block !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ececec !important;
    }
    
    /* Make sure sidebar content is visible */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Sidebar toggle button - make it visible and functional */
    button[data-testid="baseButton-header"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Ensure main content adjusts for sidebar */
    .main {
        margin-left: 280px !important;
    }
    
    /* Sidebar header */
    .sidebar-header {
        padding: 1rem;
        border-bottom: 1px solid #343541;
    }
    
    /* Search bar in sidebar */
    .sidebar-search {
        width: 100%;
        padding: 0.5rem;
        background-color: #343541;
        border: 1px solid #565869;
        border-radius: 8px;
        color: #ececec;
        font-size: 14px;
    }
    
    .sidebar-search:focus {
        outline: none;
        border-color: #565869;
    }
    
    /* Chat history items */
    .chat-history-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0.5rem;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.2s;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 14px;
    }
    
    .chat-history-item:hover {
        background-color: #343541;
    }
    
    .chat-history-item.active {
        background-color: #343541;
        border-left: 3px solid #10a37f;
    }
    
    /* New chat button */
    .new-chat-button {
        width: 100%;
        padding: 0.75rem;
        margin: 0.5rem;
        background-color: transparent;
        border: 1px solid #565869;
        border-radius: 8px;
        color: #ececec;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 14px;
    }
    
    .new-chat-button:hover {
        background-color: #343541;
        border-color: #10a37f;
    }
    
    /* User profile at bottom */
    .user-profile {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        border-top: 1px solid #343541;
        background-color: #171717;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Sidebar content area - scrollable */
    .sidebar-content {
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding-bottom: 80px;
    }
    
    /* Hide default Streamlit sidebar elements we don't need */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding: 0;
    }
    
    /* Chat input wrapper - makes columns look like one component */
    .chat-input-wrapper {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 0;
        max-width: 100%;
    }
    
    /* Remove gap between columns */
    .chat-input-wrapper [data-testid="column"] {
        padding: 0 !important;
    }
    
    /* Button styled to look like part of input */
    .chat-input-wrapper button[key="add_files_button"] {
        background-color: #40414f !important;
        border: 1px solid #565869 !important;
        border-right: none !important;
        border-radius: 12px 0 0 12px !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 20px !important;
        padding: 0 !important;
        min-width: 48px !important;
        width: 100% !important;
        height: 56px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .chat-input-wrapper button[key="add_files_button"]:hover {
        background-color: #4a4b59 !important;
        color: rgba(255, 255, 255, 1) !important;
    }
    
    /* Chat input styled to connect seamlessly */
    .chat-input-wrapper .stChatInput > div > div > div {
        background-color: #40414f !important;
        border-radius: 0 !important;
        border: 1px solid #565869 !important;
        border-left: none !important;
        border-right: 1px solid #565869 !important;
        margin: 0 !important;
    }
    
    /* Send button (if visible) */
    .chat-input-wrapper .stChatInput button {
        border-radius: 0 12px 12px 0 !important;
    }
    
    .chat-input-wrapper .stChatInput > div > div > div > textarea {
        color: white !important;
        font-size: 16px !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #565869;
        background-color: #343541;
        color: white;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #40414f;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #343541;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Welcome message */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        color: #888;
        font-size: 1rem;
    }
    
    /* Ensure messages are displayed in order */
    [data-testid="stChatMessage"] {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_qa_system(user_id: Optional[str] = None, use_biobert: bool = False):
    """
    Initialize the File QA system with optional user-specific vector database

    Args:
        user_id: Optional user ID for collection isolation
        use_biobert: If True, use BioBERT for better medical document retrieval (default: False for Cloud Run)
    """
    try:
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
            st.error("⚠️ API key not configured. Please contact administrator.")
            return None

        logger.info(f"Initializing QA system (use_biobert={use_biobert}, user_id={user_id})...")
        
        # Check if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("✅ sentence-transformers is available")
        except ImportError as e:
            logger.error(f"❌ sentence-transformers not available: {e}")
            st.error("⚠️ Embedding library not available. Please contact administrator.")
            return None
        
        # Try to initialize with BioBERT first if requested, otherwise use default
        qa_system = None
        if use_biobert:
            try:
                logger.info("Attempting to initialize with BioBERT...")
                qa_system = FileQA(
                    gemini_api_key=api_key,
                    use_biobert=True,
                    use_vector_db=True,
                    user_id=user_id,
                    simplify_medical_terms=True
                )
                # Verify embedding model is loaded
                if qa_system.rag.embedding_model is None:
                    raise ValueError("BioBERT embedding model failed to load")
                logger.info("✅ QA system initialized with BioBERT successfully")
            except Exception as biobert_error:
                logger.warning(f"BioBERT initialization failed: {biobert_error}. Falling back to default model...")
                use_biobert = False
        
        if not qa_system:
            # Use default embedding model (all-MiniLM-L6-v2)
            logger.info("Initializing with default embedding model (all-MiniLM-L6-v2)...")
            qa_system = FileQA(
                gemini_api_key=api_key,
                use_biobert=False,  # Use default model
                use_vector_db=True,
                user_id=user_id,
                simplify_medical_terms=True
            )
            # Verify embedding model is loaded
            if qa_system.rag.embedding_model is None:
                error_msg = "Default embedding model failed to load. Check logs for details."
                logger.error(error_msg)
                st.error(f"⚠️ {error_msg}")
                return None
            logger.info("✅ QA system initialized with default model successfully")
        
        return qa_system
    except Exception as e:
        logger.error(f"Failed to initialize QA system: {e}", exc_info=True)
        st.error(f"⚠️ System initialization failed: {str(e)}")
        return None


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def create_new_chat():
    """Create a new chat session"""
    st.session_state.messages = []
    st.session_state.documents_loaded = False
    st.session_state.loaded_files = []
    st.session_state.current_chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main():
    # Initialize session state
    if 'qa_system' not in st.session_state:
        # Use current_chat_id as user_id for collection isolation
        user_id = st.session_state.get('current_chat_id', None)
        st.session_state.qa_system = initialize_qa_system(user_id=user_id)

        # Check if initialization failed
        if st.session_state.qa_system is None:
            st.error("⚠️ Failed to initialize the Q&A system. Please check the logs or contact support.")
            st.stop()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'loaded_files' not in st.session_state:
        st.session_state.loaded_files = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if 'show_file_upload' not in st.session_state:
        st.session_state.show_file_upload = False
    
    if 'load_success_message' not in st.session_state:
        st.session_state.load_success_message = None
    
    # Sidebar - ChatGPT-style Chat History
    # Ensure sidebar is always visible and functional
    with st.sidebar:
        # Sidebar header - always visible
        st.markdown("# 🏥 Lab Lens")
        st.markdown("---")
        
        # New Chat button - ChatGPT style
        new_chat_clicked = st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar", type="primary")
        if new_chat_clicked:
            create_new_chat()
            st.session_state.show_file_upload = False
            # Reinitialize QA system with new chat's user_id
            st.session_state.qa_system = initialize_qa_system(user_id=st.session_state.current_chat_id)
            st.rerun()
        
        st.markdown("---")
        
        # Chat history section - always visible
        st.markdown("### 💬 Recent Chats")
        
        # Debug: Show current state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = {}
        
        # Display chat history - ChatGPT style
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            # Sort by timestamp (most recent first)
            sorted_chats = sorted(
                st.session_state.chat_history.items(),
                key=lambda x: x[1].get('timestamp', ''),
                reverse=True
            )
            
            # Display each chat
            for chat_id, chat_data in sorted_chats:
                chat_name = chat_data.get('name', chat_id)
                is_active = chat_id == st.session_state.current_chat_id
                
                # Truncate long names
                display_name = chat_name[:35] + "..." if len(chat_name) > 35 else chat_name
                
                # Create a button for each chat
                button_style = "primary" if is_active else "secondary"
                
                if st.button(
                    display_name,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type=button_style
                ):
                    # Load this chat and reinitialize QA system with this chat's collection
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = chat_data.get('messages', [])
                    st.session_state.documents_loaded = chat_data.get('documents_loaded', False)
                    st.session_state.loaded_files = chat_data.get('loaded_files', [])
                    st.session_state.show_file_upload = False
                    # Reinitialize QA system with this chat's user_id for collection isolation
                    st.session_state.qa_system = initialize_qa_system(user_id=chat_id)
                    st.rerun()
        else:
            # Show empty state
            st.info("No chat history yet. Start a conversation to see it here!")
        
        st.markdown("---")
        
        # User profile section
        st.markdown("### 👤 User")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("""
            <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 16px;">
                S
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("**Shahid Kamal**")
            st.caption("Free Plan")
    
    # Show persistent status indicator if documents are loaded (at top)
    if st.session_state.documents_loaded:
        st.success(f"📄 **Documents loaded:** {', '.join(st.session_state.loaded_files[:3])}{' ...' if len(st.session_state.loaded_files) > 3 else ''} • Ready to answer questions!")
    
    # Show success message if available
    if st.session_state.load_success_message:
        st.info(st.session_state.load_success_message)
        # Clear message after showing (so it doesn't persist forever)
        st.session_state.load_success_message = None
    
    # Main chat area - Clean ChatGPT-style
    if not st.session_state.messages:
        # Welcome screen (only show if no documents loaded or no messages)
        if not st.session_state.documents_loaded:
            st.markdown("""
            <div class="welcome-container">
                <div class="welcome-title">🏥 Lab Lens</div>
                <div class="welcome-subtitle">File Q&A System</div>
                <div class="welcome-subtitle" style="margin-top: 1rem;">Upload documents and ask questions about them</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show a welcome message for loaded documents
            st.markdown(f"""
            <div class="welcome-container">
                <div class="welcome-title">📄 Documents Ready</div>
                <div class="welcome-subtitle">Ask questions about your uploaded documents</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Display chat messages in chronological order (oldest first, newest at bottom)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        score = source.get('score', 0)
                        preview = source.get('chunk', '')[:200]
                        st.caption(f"Source {i} (relevance: {score:.3f})")
                        st.text(preview + "...")
    
    # File upload modal (only appears when + button is clicked)
    if st.session_state.show_file_upload:
        st.markdown("---")
        st.markdown("### 📁 Upload Files")
        col1, col2 = st.columns([2, 1])
        with col1:
            quick_upload = st.file_uploader(
                "Upload files",
                type=['txt', 'pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'md'],
                accept_multiple_files=True,
                help="Upload text files, PDFs, or images",
                label_visibility="collapsed",
                key="quick_upload_main"
            )
        with col2:
            quick_text = st.text_area(
                "Or paste text",
                height=100,
                help="Paste raw text content",
                label_visibility="collapsed",
                placeholder="Paste text here...",
                key="quick_text_main"
            )
        
        col_load, col_close = st.columns([1, 1])
        with col_load:
            if st.button("📥 Load Documents", type="primary", use_container_width=True, key="load_main_btn"):
                if quick_upload or (quick_text and quick_text.strip()):
                    with st.spinner("Processing documents..."):
                        try:
                            if quick_upload:
                                file_paths = []
                                for uploaded_file in quick_upload:
                                    file_path = save_uploaded_file(uploaded_file)
                                    file_paths.append(file_path)
                                
                                result = st.session_state.qa_system.load_multiple_files(file_paths)
                                if result.get('success'):
                                    st.session_state.documents_loaded = True
                                    st.session_state.loaded_files = [f.name for f in quick_upload]
                                    st.session_state.show_file_upload = False
                                    st.session_state.load_success_message = f"✅ Successfully loaded {result['num_files']} file(s) ({result.get('num_chunks', 0)} chunks). Ready to answer questions!"
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to load files: {result.get('error', 'Unknown error')}")
                                    st.session_state.load_success_message = None
                            
                            if quick_text and quick_text.strip():
                                result = st.session_state.qa_system.load_text(quick_text.strip())
                                if result.get('success'):
                                    st.session_state.documents_loaded = True
                                    if "Text Input" not in st.session_state.loaded_files:
                                        st.session_state.loaded_files.append("Text Input")
                                    st.session_state.show_file_upload = False
                                    st.session_state.load_success_message = f"✅ Successfully loaded text ({result.get('num_chunks', 0)} chunks). Ready to answer questions!"
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to load text: {result.get('error', 'Unknown error')}")
                                    st.session_state.load_success_message = None
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Please upload a file or paste text")
        
        with col_close:
            if st.button("❌ Close", use_container_width=True, key="close_main_btn"):
                st.session_state.show_file_upload = False
                st.rerun()
        
        st.markdown("---")
    
    # Add spacing at bottom for fixed input
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Fixed chat input at bottom - always visible
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
    
    # Use columns: + button, input
    col_add, col_input = st.columns([0.05, 0.95], gap="small")
    
    with col_add:
        # + button styled to connect with input
        if st.button("➕", help="Add files", key="add_files_button", use_container_width=True):
            st.session_state.show_file_upload = not st.session_state.show_file_upload
            st.rerun()
    
    with col_input:
        chat_placeholder = "Ask a question about your documents..." if st.session_state.documents_loaded else "Upload files first to ask questions..."
        
        if prompt := st.chat_input(chat_placeholder, key="chat_input_main"):
            # Check if documents are loaded
            if not st.session_state.documents_loaded:
                st.warning("⚠️ **Documents not loaded yet!** Please:\n1. Click ➕ button to upload files\n2. Click '📥 Load Documents' button to process them\n3. Then ask your question")
                st.stop()
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Check if user wants a summary (use MedicalSummarizer)
            prompt_lower = prompt.lower().strip()
            is_summary_request = any(keyword in prompt_lower for keyword in [
                'summarize', 'summary', 'summarise', 'summarise this', 
                'give me a summary', 'create a summary', 'generate summary'
            ])
            
            # Generate response
            with st.spinner("Generating summary using MedicalSummarizer..." if is_summary_request else "Thinking..."):
                try:
                    if is_summary_request:
                        # Use MedicalSummarizer for summarization
                        result = st.session_state.qa_system.summarize_document()
                        
                        if result.get('success'):
                            answer = result.get('summary', 'Summary not available')
                            sources = []  # Summarizer doesn't return sources in the same format
                        else:
                            # Fallback to Gemini if summarizer fails
                            error_msg = result.get('error', 'Unknown error')
                            logger.warning(f"Summarizer failed: {error_msg}. Falling back to Gemini.")
                            result = st.session_state.qa_system.ask_question(prompt)
                            answer = result.get('answer', 'No answer available')
                            sources = result.get('sources', [])
                    else:
                        # Use Gemini for Q&A
                        result = st.session_state.qa_system.ask_question(prompt)
                        
                        answer = result.get('answer', 'No answer available')
                        sources = result.get('sources', [])
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Save to chat history (keep existing name if chat already exists)
                    if st.session_state.current_chat_id in st.session_state.chat_history:
                        # Chat already exists - keep the existing name
                        existing_chat = st.session_state.chat_history[st.session_state.current_chat_id]
                        chat_name = existing_chat.get('name', prompt[:30] + "..." if len(prompt) > 30 else prompt)
                    else:
                        # New chat - set name from first prompt
                        chat_name = prompt[:30] + "..." if len(prompt) > 30 else prompt
                    
                    st.session_state.chat_history[st.session_state.current_chat_id] = {
                        'name': chat_name,  # Keep the same name, don't update it
                        'messages': st.session_state.messages.copy(),
                        'documents_loaded': st.session_state.documents_loaded,
                        'loaded_files': st.session_state.loaded_files.copy(),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Rerun to display new messages and scroll to bottom
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Scroll to bottom button
    st.markdown("""
    <button class="scroll-to-bottom-btn" id="scrollToBottomBtn" onclick="scrollToBottom()" title="Scroll to bottom">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 12l-7 7-7-7M19 5l-7 7-7-7"/>
        </svg>
    </button>
    """, unsafe_allow_html=True)
    
    # JavaScript for auto-scroll and scroll detection
    st.markdown("""
    <script type="text/javascript">
        let autoScrollEnabled = true;
        let userScrolledUp = false;
        let lastMessageCount = 0;
        
        function scrollToBottom() {
            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });
            autoScrollEnabled = true;
            userScrolledUp = false;
            updateScrollButton();
        }
        
        function updateScrollButton() {
            const btn = document.getElementById('scrollToBottomBtn');
            if (!btn) return;
            
            const scrollHeight = document.documentElement.scrollHeight;
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const clientHeight = document.documentElement.clientHeight;
            const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
            
            // Show button if user is more than 200px from bottom
            if (distanceFromBottom > 200) {
                btn.classList.add('visible');
            } else {
                btn.classList.remove('visible');
            }
        }
        
        function checkForNewMessages() {
            const currentMessageCount = document.querySelectorAll('[data-testid="stChatMessage"]').length;
            
            // If new messages were added, auto-scroll if user hasn't scrolled up
            if (currentMessageCount > lastMessageCount && !userScrolledUp) {
                setTimeout(() => {
                    scrollToBottom();
                }, 100);
            }
            
            lastMessageCount = currentMessageCount;
        }
        
        // Auto-scroll on page load
        window.addEventListener('load', function() {
            setTimeout(() => {
                scrollToBottom();
                checkForNewMessages();
            }, 300);
        });
        
        // Detect user scrolling
        let scrollTimeout;
        window.addEventListener('scroll', function() {
            clearTimeout(scrollTimeout);
            
            const scrollHeight = document.documentElement.scrollHeight;
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const clientHeight = document.documentElement.clientHeight;
            const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
            
            // If user is more than 100px from bottom, they've scrolled up
            if (distanceFromBottom > 100) {
                userScrolledUp = true;
                autoScrollEnabled = false;
            } else {
                // User is near bottom, re-enable auto-scroll
                userScrolledUp = false;
                autoScrollEnabled = true;
            }
            
            updateScrollButton();
            
            // Check for new messages after scroll settles
            scrollTimeout = setTimeout(checkForNewMessages, 100);
        });
        
        // Monitor for new messages (MutationObserver)
        const observer = new MutationObserver(function(mutations) {
            checkForNewMessages();
        });
        
        // Start observing when DOM is ready
        if (document.body) {
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        } else {
            document.addEventListener('DOMContentLoaded', function() {
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
            });
        }
        
        // Periodic check for new messages (fallback)
        setInterval(checkForNewMessages, 500);
    </script>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
