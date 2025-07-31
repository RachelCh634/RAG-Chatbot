import streamlit as st
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

def get_bot_response_from_api(user_message):
    """Get response from API endpoint"""
    api_url = "http://localhost:8000/chat"
    try:
        payload = {"query": user_message}
        response = requests.post(api_url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "⚠️ No answer returned.")
    except requests.exceptions.ConnectionError:
        return "Cannot connect to server. Please check if the backend is running."
    except requests.exceptions.Timeout:
        return "⏱️ Request timeout. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"Server error: {e.response.status_code}"
    except Exception as e:
        return f"Unexpected error: {str(e)[:100]}"

def configure_page():
    """Set up page configuration"""
    try:
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="centered",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        .main {
            padding-top: 1rem;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .chat-container {
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            background-color: #ffffff;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 600px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .upload-container {
            border: 2px dashed #6366f1;
            border-radius: 12px;
            background-color: #f8fafc;
            margin: 1rem 0;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .chat-header {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            padding: 1rem;
            text-align: center;
            font-weight: bold;
            border-bottom: 1px solid #e5e7eb;
            flex-shrink: 0;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            background-color: #fafafa;
            min-height: 0;
            max-height: 500px;
        }
        
        .user-message {
            background-color: #f7f7f8;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #10a37f;
            max-width: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
        
        .bot-message {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #6366f1;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            max-width: 100%;
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
        
        .message-time {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }
        
        .input-container {
            background-color: white;
            padding: 1rem;
            border-top: 1px solid #e5e7eb;
            flex-shrink: 0;
        }
        
        .welcome-message {
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .pdf-info {
            background-color: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #0c4a6e;
        }
        
        .stTextInput > div > div > input {
            border-radius: 4px;
            padding: 0.5rem 0.75rem;
        }
        
        .stTextInput > div > div > input:focus {
            outline: none;
        }

        .sidebar-content {
            padding: 1rem;
        }
        
        .chat-messages::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        
        .title {
            text-align: center;
            color: #374151;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "messages": [],
        "chat_history": [],
        "input_key": 0,
        "pdf_uploaded": False,
        "pdf_content": "",
        "pdf_filename": "",
        "processing_message": False,
        "show_history_modal": False,
        "last_input": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def safe_api_call(func, error_message="Operation failed"):
    """Safely execute API calls with error handling"""
    try:
        return func()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to server")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout")
        return None
    except Exception as e:
        st.error(f" {error_message}: {str(e)[:50]}")
        return None

def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.header("⚙️ Controls")
        
        st.subheader("💬 Chat Management")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.input_key += 1
            st.session_state.processing_message = False
            st.rerun()
        
        st.subheader("📄 Document Management")
        if st.session_state.get("pdf_uploaded", False):
            if st.button("📤 Upload New PDF"):
                def clear_vectors():
                    response = requests.post("http://localhost:8000/clear_all_vectors", timeout=60)
                    response.raise_for_status()
                    return response.json()
                
                result = safe_api_call(clear_vectors, "Failed to clear vectors")
                if result:
                    st.session_state.pdf_uploaded = False
                    st.session_state.pdf_content = ""
                    st.session_state.pdf_filename = ""
                    st.session_state.messages = []
                    st.session_state.processing_message = False
                    st.session_state.input_key += 1
                    st.success("Ready for new document")
                    st.rerun()
        else:
            st.info("📄 Upload a PDF to start chatting")
        
        st.subheader("🔧 Server Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧠 Clear Memory"):
                def clear_memory():
                    response = requests.post("http://localhost:8000/clear-memory", timeout=60)
                    response.raise_for_status()
                    return response.json()
                
                result = safe_api_call(clear_memory, "Failed to clear memory")
                if result:
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.session_state.input_key += 1
                    st.session_state.processing_message = False
                    st.success("Memory cleared")
                    st.rerun()
        
        with col2:
            if st.button("📜 Show History"):
                st.session_state.show_history_modal = True
                st.rerun()

        st.markdown("---")
        if st.button("🔴 Complete Reset"):
            def reset_all():
                requests.post("http://localhost:8000/clear-memory", timeout=60)
                requests.post("http://localhost:8000/clear_all_vectors", timeout=60)
                return True
            
            result = safe_api_call(reset_all, "Reset failed")
            if result:
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.pdf_uploaded = False
                st.session_state.pdf_content = ""
                st.session_state.pdf_filename = ""
                st.session_state.processing_message = False
                st.session_state.input_key += 1
                st.success("Complete reset done")
                st.rerun()

def display_history_modal():
    """Display conversation history modal"""
    if not st.session_state.get("show_history_modal", False):
        return
        
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0;">🧾 Server History</h3>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        if st.button("Close", key="close_history"):
            st.session_state.show_history_modal = False
            st.rerun()
    
    with st.spinner("Loading history..."):
        def get_history():
            response = requests.get("http://localhost:8000/conversation-history", timeout=60)
            response.raise_for_status()
            return response.json()
        
        response = safe_api_call(get_history, "Failed to load history")
        
        if response and response.get("status") == "success" and "history" in response:
            history = response["history"]
            if history:
                for i, item in enumerate(history, 1):
                    question = item.get('question', 'N/A')
                    answer = item.get('answer', 'N/A')
                    
                    st.markdown(f"""
                    <div style="
                        background-color: #f7f7f8;
                        padding: 1rem;
                        border-radius: 8px;
                        margin: 1rem 0;
                        border-left: 4px solid #10a37f;
                    ">
                        <strong>👤 Question {i}:</strong><br>
                        {question}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="
                        background-color: #ffffff;
                        padding: 1rem;
                        border-radius: 8px;
                        margin: 1rem 0;
                        border-left: 4px solid #6366f1;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    ">
                        <strong>🤖 Answer:</strong><br>
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if i < len(history):
                        st.markdown("<hr style='margin: 2rem 0; border-color: #e5e7eb;'>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    text-align: center;
                    padding: 3rem;
                    background-color: #f8fafc;
                    border: 2px dashed #cbd5e1;
                    border-radius: 12px;
                    color: #64748b;
                    margin: 1rem 0;
                    width: 100%;
                ">
                    <h3>📭 No History Available</h3>
                    <p>No conversations have been saved on the server yet</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem;
                background-color: #fef2f2;
                border: 2px solid #fca5a5;
                border-radius: 12px;
                color: #dc2626;
                margin: 2rem 0;
                width: 100%;
            ">
                <h3>Error Loading History</h3>
                <p>Unable to load history from the server</p>
            </div>
            """, unsafe_allow_html=True)

def handle_pdf_upload():
    """Handle PDF file upload"""
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze and chat about"
    )
    
    if uploaded_file is None:
        return
        
    if uploaded_file.size > 200 * 1024 * 1024:
        st.error("File too large. Please upload a PDF smaller than 200MB.")
        return
    
    if uploaded_file.size == 0:
        st.error("File is empty. Please upload a valid PDF file.")
        return
        
    with st.spinner("Uploading PDF to server..."):
        try:
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            if len(file_content) == 0:
                st.error("File content is empty.")
                return
            
            uploaded_file.seek(0)
            files = {
                'file': (uploaded_file.name, uploaded_file, 'application/pdf')
            }
            
            response = requests.post(
                "http://localhost:8000/upload-pdf", 
                files=files,
                timeout=1000
            )
            
            if response.status_code == 400:
                try:
                    error_detail = response.json().get("detail", response.text)
                except:
                    error_detail = response.text
                st.error(f"Upload failed: {error_detail}")
                return
            
            response.raise_for_status()
            data = response.json()

            st.session_state.pdf_uploaded = True
            st.session_state.pdf_filename = data.get("filename", uploaded_file.name)
            st.session_state.pdf_content = "Uploaded to vector DB"

            st.success(f"PDF '{uploaded_file.name}' uploaded successfully!")
            st.markdown(f"""
            <div class="pdf-info">
                <strong>📄 Document Info:</strong><br>
                <strong>Filename:</strong> {st.session_state.pdf_filename}<br>
                <strong>Status:</strong> {data.get("message", "")}
            </div>
            """, unsafe_allow_html=True)
            
            st.rerun()
            
        except requests.exceptions.ConnectionError:
            st.error(" Cannot connect to server. Make sure your backend server is running on http://localhost:8000")
        except requests.exceptions.Timeout:
            st.error(" Upload timeout. The file might be too large or the server is busy.")
        except requests.exceptions.HTTPError as e:
            st.error(f" HTTP error during upload: {e}")
        except Exception as e:
            st.error(f" Upload failed: {str(e)}")

def display_pdf_info():
    """Display current PDF information"""
    st.markdown(f"""
    <div class="pdf-info">
        <strong>📄 Current Document:</strong> {st.session_state.pdf_filename}<br>
        <strong>Status:</strong> Ready for questions
    </div>
    """, unsafe_allow_html=True)

def display_welcome_message():
    """Display welcome message for new chat"""
    st.markdown(f"""
    <div class="welcome-message">
        <h3>Hello! I'm ready to help you with your document</h3>
        <p>I've analyzed your PDF: <strong>{st.session_state.pdf_filename}</strong></p>
        <p>Ask me anything about the content!</p>
    </div>
    """, unsafe_allow_html=True)

def display_chat_messages():
    """Display chat conversation messages exactly like history"""
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            content = str(message.get("content", ""))
            timestamp = message.get("timestamp", "")
            
            if message["role"] == "user":
                st.markdown(f"""
                <div style="
                    background-color: #f7f7f8;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    border-left: 4px solid #10a37f;
                ">
                    <strong>👤 You:</strong><br>
                    {content}
                    <div style="
                        font-size: 0.8rem;
                        color: #6b7280;
                        margin-top: 0.5rem;
                    ">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background-color: #ffffff;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    border-left: 4px solid #6366f1;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                ">
                    <strong>🤖 Assistant:</strong><br>
                    {content}
                    <div style="
                        font-size: 0.8rem;
                        color: #6b7280;
                        margin-top: 0.5rem;
                    ">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)

def handle_user_input():
    """Handle user input and send button"""
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask about your document...",
            key=f"user_input_{st.session_state.input_key}",
            placeholder="🤖 Processing..." if st.session_state.processing_message else "What would you like to know about this document?",
            label_visibility="collapsed",
            disabled=st.session_state.processing_message
        )
    
    with col2:
        send_button = st.button(
            "🤖" if st.session_state.processing_message else "Send", 
            use_container_width=True,
            disabled=st.session_state.processing_message
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return user_input, send_button

def process_message(user_input):
    """Process user message and add to conversation"""
    if not user_input or not user_input.strip():
        return
        
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    st.session_state.processing_message = True
    st.session_state.input_key += 1
    st.rerun()

def get_bot_response_and_update():
    """Get bot response and update conversation"""
    if not st.session_state.processing_message or not st.session_state.messages:
        return
        
    last_user_message = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break
    
    if not last_user_message:
        st.session_state.processing_message = False
        return
        
    display_chat_messages()
    
    with st.spinner("🤖 Thinking..."):
        bot_response = get_bot_response_from_api(last_user_message)
    
    if bot_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": str(bot_response),
            "timestamp": datetime.now().strftime("%H:%M")
        })
    
    st.session_state.processing_message = False
    st.rerun()

def render_footer():
    """Render page footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;'>
        <p>🤖 RAG Chatbot | Built with Streamlit | PDF Analysis Mode</p>
    </div>
    """, unsafe_allow_html=True)

def is_server_ready():
    api_url = "http://localhost:8000/healthcheck"
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200 and response.json().get("status") == "ready":
            return True
    except requests.RequestException:
        pass
    return False

def main():
    if not is_server_ready():
        st.error("Server is not ready yet")
        st_autorefresh(interval=3000, key="server_check")
        st.stop()
                   
    configure_page()
    apply_custom_css()
    initialize_session_state()
    
    st.markdown("<h1 class='title'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)
    
    render_sidebar()
    
    if st.session_state.get("show_history_modal", False):
        display_history_modal()
        return  
    
    if not st.session_state.pdf_uploaded:
        handle_pdf_upload()
    else:
        display_pdf_info()
        
        if st.session_state.processing_message:
            get_bot_response_and_update()
        elif not st.session_state.messages:
            display_welcome_message()
        else:
            display_chat_messages()
        
        user_input, send_button = handle_user_input()
        
        if not st.session_state.processing_message:
            should_send = False
            
            if send_button and user_input and user_input.strip():
                should_send = True
            elif user_input and user_input.strip():
                last_input = st.session_state.get('last_input')
                if last_input != user_input:
                    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_input:
                        should_send = True
            
            if should_send:
                st.session_state.last_input = user_input
                process_message(user_input)
    
    render_footer()

if __name__ == "__main__":
    main()