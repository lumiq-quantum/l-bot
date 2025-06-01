import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
import re
import shutil

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8090")

# Page configuration
st.set_page_config(
    page_title="LEO Document Extracter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage .stChatContent {
        font-size: 16px;
    }
    .file-info-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 5px 10px;
        margin-bottom: 5px;
        font-size: 12px;
    }
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-info {
        font-size: 12px;
        color: #888;
        margin-bottom: 20px;
    }
    .extraction-result {
        background-color: #f8f9fa;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .document-icon {
        font-size: 24px;
        margin-right: 10px;
    }
    .sidebar-title {
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# Helper functions
def fetch_chat_sessions():
    """Fetch all chat sessions from the backend"""
    try:
        response = requests.get(f"{API_URL}/chat/sessions")
        if response.status_code == 200:
            sessions = response.json()
            st.session_state.chat_sessions = sessions
            return sessions
        else:
            st.error(f"Failed to fetch chat sessions: {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
        return []


def create_new_chat():
    """Create a new chat session"""
    try:
        response = requests.post(f"{API_URL}/chat/new")
        if response.status_code == 200:
            session = response.json()
            st.session_state.current_session_id = session["id"]
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            fetch_chat_sessions()  # Refresh the list
            return session
        else:
            st.error(f"Failed to create new chat: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
        return None


def load_chat_history(session_id):
    """Load chat history for a session"""
    try:
        response = requests.get(f"{API_URL}/chat/{session_id}/history")
        if response.status_code == 200:
            history = response.json()
            st.session_state.messages = history["messages"]
            st.session_state.uploaded_files = []
            return history
        else:
            st.error(f"Failed to load chat history: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
        return None


def delete_chat_session(session_id):
    """Delete a chat session"""
    try:
        response = requests.delete(f"{API_URL}/chat/{session_id}")
        if response.status_code == 200:
            if st.session_state.current_session_id == session_id:
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.session_state.uploaded_files = []
            fetch_chat_sessions()  # Refresh the list
            return True
        else:
            st.error(f"Failed to delete chat session: {response.text}")
            return False
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
        return False


def send_message(session_id, message, uploaded_file=None):
    """Send a message to the backend and process response"""
    try:
        # Validate the message to prevent empty submissions
        if not message or message.strip() == "":
            st.error("Please enter a message before sending.")
            return False
            
        # Track the uploaded file for display purposes
        if uploaded_file:
            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "size": uploaded_file.size
            })
        
        # Add the user message to local state immediately
        file_info = ""
        if uploaded_file:
            file_info = f"📎 *Uploaded file: {uploaded_file.name}*"
            
        user_message = {
            "id": f"temp-{int(time.time())}",
            "role": "user",
            "content": f"{message}\n\n{file_info}" if file_info else message,
            "timestamp": datetime.now().isoformat(),
            "has_file": uploaded_file is not None
        }
        st.session_state.messages.append(user_message)
        
        # Display user message in the chat
        st.chat_message("user").write(user_message["content"])
        
        # Create a placeholder for the assistant's message with a loading spinner
        with st.spinner("Getting response..."):
            # Prepare data for the request - using FormData format
            files = {}
            
            if uploaded_file is not None:
                # Reset the file position to the beginning
                uploaded_file.seek(0)
                file_content = uploaded_file.read()
                files = {
                    "file": (uploaded_file.name, file_content, uploaded_file.type)
                }
            
            # Message must be sent as form data, not JSON
            data = {"message": message}
            
            # Make request to backend
            url = f"{API_URL}/chat/{session_id}/message"
            
            response = requests.post(
                url,
                data=data,  # Send as form data
                files=files
            )
            
            if response.status_code != 200:
                st.error(f"Error from server: {response.text}")
                return False
            
            # Process the JSON response
            assistant_response = response.json()
            assistant_content = assistant_response.get("content", "")
            
            # Add assistant response to local state
            assistant_message = {
                "id": assistant_response.get("id", f"temp-assistant-{int(time.time())}"),
                "role": "model",
                "content": assistant_content,
                "timestamp": assistant_response.get("timestamp", datetime.now().isoformat())
            }
            st.session_state.messages.append(assistant_message)
            
            # Display assistant message in the chat
            st.chat_message("assistant").markdown(assistant_content)
            
            # Clean up temporary files after receiving response from Gemini
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_uploads")
            clean_up_temp_files(temp_dir)
            
            # Make sure we reload history from the server to synchronize
            try:
                load_chat_history(session_id)
            except Exception as e:
                # If loading history fails, we still have our local state
                pass
                
            return True
    
    except Exception as e:
        st.error(f"Failed to send message: {str(e)}")
        return False


def send_message_with_files(session_id, message, uploaded_files):
    """Send a message with multiple files to the backend and process response"""
    try:
        # Validate the message to prevent empty submissions
        if not message or message.strip() == "":
            st.error("Please enter a message before sending.")
            return False
            
        # Add the user message to local state immediately
        file_info = ""
        if uploaded_files:
            file_names = [f.name for f in uploaded_files]
            if len(file_names) == 1:
                file_info = f"📎 *Uploaded file: {file_names[0]}*"
            else:
                file_info = f"📎 *Uploaded {len(file_names)} files:* \n"
                for i, name in enumerate(file_names):
                    file_info += f"*   {i+1}. {name}\n"
            
        user_message = {
            "id": f"temp-{int(time.time())}",
            "role": "user",
            "content": f"{message}\n\n{file_info}" if file_info else message,
            "timestamp": datetime.now().isoformat(),
            "has_file": bool(uploaded_files)
        }
        st.session_state.messages.append(user_message)
        
        # Display user message in the chat
        st.chat_message("user").write(user_message["content"])
        
        # Create a placeholder for the assistant's message with a loading spinner
        with st.spinner("Processing documents and generating response..."):
            # Prepare data for the request - using FormData format
            files_dict = {}
            
            if uploaded_files:
                for i, file in enumerate(uploaded_files):
                    # Reset the file position to the beginning
                    file.seek(0)
                    file_content = file.read()
                    # Use unique field names for each file with index to preserve multiple files
                    files_dict[f"files[{i}]"] = (file.name, file_content, file.type)
            
            # Message must be sent as form data, not JSON
            data = {"message": message}
            
            # Make request to backend
            url = f"{API_URL}/chat/{session_id}/message"
            
            response = requests.post(
                url,
                data=data,
                files=files_dict
            )
            
            if response.status_code != 200:
                st.error(f"Error from server: {response.text}")
                return False
            
            # Process the JSON response
            assistant_response = response.json()
            assistant_content = assistant_response.get("content", "")
            num_files_processed = assistant_response.get("num_files_processed", 0)
            
            if num_files_processed > 0:
                st.success(f"Successfully processed {num_files_processed} document(s)")
            
            # Add assistant response to local state
            assistant_message = {
                "id": assistant_response.get("id", f"temp-assistant-{int(time.time())}"),
                "role": "model",
                "content": assistant_content,
                "timestamp": assistant_response.get("timestamp", datetime.now().isoformat())
            }
            st.session_state.messages.append(assistant_message)
            
            # Display assistant message in the chat
            st.chat_message("assistant").markdown(assistant_content)
            
            # Clean up temporary files after receiving response from Gemini
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_uploads")
            clean_up_temp_files(temp_dir)
            
            # Make sure we reload history from the server to synchronize
            try:
                load_chat_history(session_id)
            except Exception as e:
                # If loading history fails, we still have our local state
                pass
                
            return True
    
    except Exception as e:
        st.error(f"Failed to send message: {str(e)}")
        return False


def format_file_size(size_bytes):
    """Format file size in a human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def clean_up_temp_files(temp_dir):
    """Clean up temporary files after receiving URI from Gemini"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            # st.info(f"Temporary files in {temp_dir} have been cleaned up.")
    except Exception as e:
        st.error(f"Failed to clean up temporary files: {str(e)}")


# UI Components
def sidebar():
    """Render the sidebar with chat sessions"""
    with st.sidebar:
        st.markdown('<div class="sidebar-title"><span class="document-icon">📄</span><h1>Document Sessions</h1></div>', unsafe_allow_html=True)
        
        # New chat button
        if st.button("New Extraction", use_container_width=True, type="primary"):
            create_new_chat()
        
        st.divider()
        
        # Refresh button for sessions
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Previous Sessions")
        with col2:
            if st.button("🔄", help="Refresh Sessions"):
                fetch_chat_sessions()
        
        # No sessions message
        if not st.session_state.chat_sessions:
            st.info("No previous sessions. Start a new extraction!")
        
        # Display chat sessions
        for session in st.session_state.chat_sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                session_title = session.get("title", "New Session")
                if len(session_title) > 25:
                    session_title = session_title[:22] + "..."
                    
                # Add visual indicator for current session
                if st.session_state.current_session_id == session["id"]:
                    session_title = f"▶️ {session_title}"
                else:
                    session_title = f"📄 {session_title}"
                    
                if st.button(
                    session_title,
                    key=f"session_{session['id']}",
                    use_container_width=True
                ):
                    st.session_state.current_session_id = session["id"]
                    load_chat_history(session["id"])
            
            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_{session['id']}",
                    help="Delete this session"
                ):
                    if st.session_state.current_session_id == session["id"]:
                        with st.spinner("Deleting current session..."):
                            delete_chat_session(session["id"])
                            st.success("Session deleted!")
                    else:
                        delete_chat_session(session["id"])
        
        # Information about the model
        st.divider()
        st.caption("🤖 Powered by AI Document Extraction")


def chat_interface():
    """Render the main chat interface"""
    # Header area
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📄 LEO Document Extracter")
    with col2:
        st.markdown('<div class="app-info">Powered by AI</div>', unsafe_allow_html=True)
    
    # Display messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                if msg["content"]:  # Only display if there's content
                    st.chat_message("assistant").markdown(msg["content"])
    
    # Input area
    if st.session_state.current_session_id:
        # Help text - show document extraction guidance
        st.info("""
        This tool analyzes documents to extract structured information, text, and data.
        
        1. Upload one or more documents (PDF, Images, Excel, etc.)
        2. Ask specific questions about the document content
        3. Request extraction of document data, text sections, or specific data points
        """)
        
        # File uploader with broader format support and multiple file selection
        uploaded_files = st.file_uploader(
            "Upload documents for extraction (PDF,  Images)",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.current_session_id}"
        )
        
        # Display information about the uploaded files
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} document(s) selected:**")
            for i, file in enumerate(uploaded_files):
                st.markdown(f"""
                <div class="file-info-box">
                    📎 File {i+1}: <b>{file.name}</b> | 
                    Type: {file.type} | 
                    Size: {format_file_size(file.size)}
                </div>
                """, unsafe_allow_html=True)
                
            if st.button("Clear Files", key=f"clear_files_{st.session_state.current_session_id}"):
                st.session_state.uploaded_files = []
                st.rerun()
        
        # Help text for user with examples
        if uploaded_files:
            st.markdown("""
            <div class="extraction-result">
            💡 <b>Example queries with multiple documents:</b>
            <ul>
                <li>"Extract all tables from these documents"</li>
                <li>"Compare the information across these documents"</li>
                <li>"Summarize the key points from all documents"</li>
                <li>"Find differences between the first and second document"</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input 
        prompt = st.chat_input(
            "Ask about document contents or request extraction..."
        )
        if prompt:
            # Trim the prompt to remove leading/trailing whitespace
            prompt = prompt.strip()
            if prompt:  # Only send if not empty after trimming
                send_message_with_files(
                    st.session_state.current_session_id,
                    prompt,
                    uploaded_files
                )
    else:
        # No active session
        st.info("👈 Start a new document extraction session or select an existing one.")
        if st.button("Start New Extraction", type="primary"):
            create_new_chat()


# Main app
def main():
    # Load initial data
    if not st.session_state.chat_sessions:
        fetch_chat_sessions()
    
    # If no current session but chat sessions exist, select the first one
    if not st.session_state.current_session_id and st.session_state.chat_sessions:
        st.session_state.current_session_id = st.session_state.chat_sessions[0]["id"]
        load_chat_history(st.session_state.current_session_id)
    
    # Render sidebar with chat sessions
    sidebar()
    
    # Render main chat interface
    chat_interface()


if __name__ == "__main__":
    main()