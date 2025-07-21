import streamlit as st
from datetime import datetime
import PyPDF2
from io import BytesIO
import requests

def get_bot_response_from_api(user_message):
    api_url = "http://localhost:8000/chat"
    try:
        payload = {"query": user_message}
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "⚠️ No answer returned.")
    except Exception as e:
        return f"❌ API error: {str(e)}"

def configure_page():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
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
        }
        
        .user-message {
            background-color: #f7f7f8;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #10a37f;
            max-width: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .bot-message {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #6366f1;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            max-width: 100%;
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = ""
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = ""

def extract_pdf_text(pdf_file):
    try:
        pdf_file.seek(0)  # Reset file pointer to beginning
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.header("⚙️ Settings")
        
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.input_key += 1
            st.rerun()
        
        if st.button("🧠 Clear Memory"):
            result = clear_memory_api()
            st.success(result)

        if st.button("🧹 Clear All Vectors"):
            result = clear_all_vectors_api()
            st.success(result)

        if st.button("📜 Show History"):
            response = get_conversation_history_api()
            if response.get("status") == "success" and "history" in response:
                history = response["history"]
                if history:
                    st.markdown("### 🧾 Server-side History:")
                    for item in history:
                        user = item.get('user', 'N/A')
                        bot = item.get('bot', 'N/A')
                        st.markdown(f"- **User:** {user}")
                        st.markdown(f"  **Bot:** {bot}")
                        st.markdown("---")
                else:
                    st.info("No history found.")
            else:
                st.error("Failed to fetch history from server.")

        if st.button("🔴 Reset Everything"):
            reset_everything()
            st.rerun()

        if st.session_state.pdf_uploaded:
            if st.button("📄 Upload New PDF"):
                st.session_state.pdf_uploaded = False
                st.session_state.pdf_content = ""
                st.session_state.pdf_filename = ""
                st.session_state.messages = []
                st.session_state.input_key += 1
                st.rerun()
        
        st.markdown("---")
        st.write(f"Messages in chat: {len(st.session_state.messages)}")
        if st.session_state.pdf_uploaded:
            st.write(f"PDF: {st.session_state.pdf_filename}")
            st.write(f"Characters: {len(st.session_state.pdf_content):,}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def handle_pdf_upload():
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze and chat about"
    )
    
    if uploaded_file is not None:
        with st.spinner("📤 Uploading PDF to server..."):
            try:
                files = {'file': (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post("http://localhost:8000/upload-pdf", files=files)
                response.raise_for_status()
                data = response.json()

                st.session_state.pdf_uploaded = True
                st.session_state.pdf_filename = data.get("filename", uploaded_file.name)
                st.session_state.pdf_content = "Uploaded to vector DB ✅"

                st.success(f"PDF '{uploaded_file.name}' uploaded successfully!")
                st.markdown(f"""
                <div class="pdf-info">
                    <strong>📄 Document Info:</strong><br>
                    <strong>Filename:</strong> {st.session_state.pdf_filename}<br>
                    <strong>Status:</strong> {data.get("message", "")}
                </div>
                """, unsafe_allow_html=True)
                
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

def display_pdf_info():
    st.markdown(f"""
    <div class="pdf-info">
        <strong>📄 Current Document:</strong> {st.session_state.pdf_filename}<br>
        <strong>Characters:</strong> {len(st.session_state.pdf_content):,}
    </div>
    """, unsafe_allow_html=True)

def display_welcome_message():
    st.markdown(f"""
    <div class="welcome-message">
        <h3>Hello! I'm ready to help you with your document</h3>
        <p>I've analyzed your PDF: <strong>{st.session_state.pdf_filename}</strong></p>
        <p>Ask me anything about the content!</p>
    </div>
    """, unsafe_allow_html=True)

def display_chat_messages():
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
                <div class="message-time">{message.get("timestamp", "")}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
                <div class="message-time">{message.get("timestamp", "")}</div>
            </div>
            """, unsafe_allow_html=True)

def scroll_to_bottom():
    st.markdown("""
    <script>
        setTimeout(function() {
            var chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }, 100);
    </script>
    """, unsafe_allow_html=True)

def handle_user_input():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask about your document...",
            key=f"user_input_{st.session_state.input_key}",
            placeholder="What would you like to know about this document?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    return user_input, send_button

def clear_memory_api():
    try:
        response = requests.post("http://localhost:8000/clear-memory")
        response.raise_for_status()
        return response.json().get("message", "✅ Memory cleared")
    except Exception as e:
        return f"❌ Failed to clear memory: {str(e)}"

def clear_all_vectors_api():
    try:
        response = requests.post("http://localhost:8000/clear_all_vectors")
        response.raise_for_status()
        return response.json().get("message", "✅ All vectors cleared")
    except Exception as e:
        return f"❌ Failed to clear vectors: {str(e)}"

def get_conversation_history_api():
    try:
        response = requests.get("http://localhost:8000/conversation-history")
        response.raise_for_status()
        return response.json() 
    except Exception as e:
        st.error(f"❌ Failed to fetch history: {str(e)}")
        return []

def reset_everything():
    msg1 = clear_memory_api()
    msg2 = clear_all_vectors_api()
    
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.pdf_uploaded = False
    st.session_state.pdf_content = ""
    st.session_state.pdf_filename = ""
    st.session_state.input_key += 1

    st.success("✅ Everything has been reset.")

def process_message(user_input):
    timestamp = datetime.now().strftime("%H:%M")
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    with st.spinner("🤖 Thinking..."):
        bot_response = get_bot_response_from_api(user_input)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    st.session_state.input_key += 1
    st.rerun()

def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;'>
        <p>🤖 RAG Chatbot | Built with Streamlit | PDF Analysis Mode</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    configure_page()
    apply_custom_css()
    initialize_session_state()
    
    st.markdown("<h1 class='title'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)
    
    render_sidebar()
    
    if not st.session_state.pdf_uploaded:
        handle_pdf_upload()
    else:
        display_pdf_info()
        
        if not st.session_state.messages:
            display_welcome_message()
        else:
            display_chat_messages()
        
        user_input, send_button = handle_user_input()
        
        if (send_button or user_input) and user_input.strip():
            process_message(user_input)
        
        if st.session_state.messages:
            scroll_to_bottom()
    
    render_footer()

if __name__ == "__main__":
    main()