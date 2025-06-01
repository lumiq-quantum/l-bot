import os
import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import pathlib

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel # Keep Pydantic BaseModel if needed elsewhere
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the project root directory (one level up from backend/)
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import necessary parts from google-generativeai
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.exc import SQLAlchemyError

# Configuration
# --- Load API keys and configuration from environment variables ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "":
    print("Warning: GOOGLE_API_KEY environment variable not set.")
    # raise ValueError("GOOGLE_API_KEY environment variable is not set") # Uncomment to enforce key

# System prompt (remains the same)
SYSTEM_PROMPT = (
    "You are an expert document reading assistant and you answer questions based on the content of the documents provided by the user. "
)

# --- Configure Google Generative AI ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel("gemini-2.5-pro-preview-03-25") 
    print("Gemini Model configured.")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    raise

# --- Database setup ---
try:
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("Warning: DATABASE_URL environment variable not set. Database functionality may not work.")
        
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print("Database engine created.")
except Exception as e:
    print(f"Error creating database engine: {e}")
    raise

# --- Database models ---
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role = Column(String, nullable=False)  # 'user' or 'model' (Gemini uses 'model')
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_uri = Column(String, nullable=True)
    file_mime_type = Column(String, nullable=True)
    session = relationship("ChatSession", back_populates="messages")

# --- Create tables ---
try:
    print("Attempting to create database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables should be created/verified.")
except Exception as e:
    print(f"Error creating database tables: {e}")

# --- FastAPI app ---
app = FastAPI(title="Gemini Chatbot API with File API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency for DB Session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic models for API responses ---
class MessageResponse(BaseModel):
    id: str
    role: str
    content: Any
    timestamp: datetime
    file_uri: Optional[str] = None
    file_mime_type: Optional[str] = None

class SessionInfo(BaseModel):
     id: str
     title: Optional[str] = None
     created_at: datetime

class SessionHistoryResponse(BaseModel):
    session: SessionInfo
    messages: List[MessageResponse]

class SessionListItem(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: datetime
    last_message: Optional[str] = None

# --- Helper Function to build history for Gemini ---
def build_gemini_history(db_messages: List[ChatMessage]) -> tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    history = []
    most_recent_file_uri = None
    most_recent_file_mime_type = None

    for msg in db_messages:
        role = 'user' if msg.role == 'user' else 'model'
        parts = [msg.content]

        if msg.file_uri:
            most_recent_file_uri = msg.file_uri
            most_recent_file_mime_type = msg.file_mime_type

        history.append({"role": role, "parts": parts})

    return history, most_recent_file_uri, most_recent_file_mime_type

# --- Modified Generator Function ---
async def generate_response(
    prompt_text: str,
    file_uri: Optional[str] = None,
    file_mime_type: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None
):
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
    }
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    try:
        print(f"Starting chat generation. History length: {len(history) if history else 0}")
        print(f"File URI for this turn: {file_uri}")

        # Build message content using SDK-agnostic structure
        message_content = []
        
        # Always include text prompt
        message_content.append({'text': prompt_text})
        
        # Add file if exists
        if file_uri and file_mime_type:
            print(f"Adding file from URI: {file_uri}")
            message_content.append({
                'file_data': {
                    'mime_type': file_mime_type,
                    'file_uri': file_uri
                }
            })

        # Start chat with simplified history structure
        chat = model.start_chat(history=history or [])
        
        # Send message with explicit content structure
        response = chat.send_message(
            {'parts': message_content},
            stream=False,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Extract text from response
        full_response = ""
        if hasattr(response, 'text'):
            full_response = response.text
        elif hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'text'):
                    full_response += part.text

        print(f"Generation complete. Response length: {len(full_response)}")
        return full_response

    except Exception as e:
        print(f"Error during Gemini generation: {e}")
        raise

# --- API Endpoints ---

@app.post("/chat/new", response_model=SessionInfo)
async def create_new_session(db: Session = Depends(get_db)):
    try:
        session = ChatSession()
        db.add(session)
        db.commit()
        db.refresh(session)
        print(f"Created new session: {session.id}")
        return {"id": session.id, "title": session.title, "created_at": session.created_at}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error creating session: {e}")
        raise HTTPException(status_code=500, detail="Database error creating session")
    except Exception as e:
        print(f"Unexpected error creating session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/{session_id}/message")
async def send_message(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        form_data = await request.form()
        files_list = []
        
        # Extract all file fields from the request (files[0], files[1], etc.)
        for key in form_data.keys():
            if key.startswith("files[") and key.endswith("]"):
                files_list.append(form_data[key])
        
        # Also handle any files sent with the traditional 'files' field
        if "files" in form_data:
            if isinstance(form_data["files"], list):
                files_list.extend(form_data["files"])
            else:
                files_list.append(form_data["files"])
        
        # Handle single file case with 'file' field
        if "file" in form_data:
            files_list.append(form_data["file"])
        
        # Extract message
        message = form_data.get("message", "")
            
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        uploaded_file_uris = []
        uploaded_file_mime_types = []
        file_names = []

        if files_list:
            allowed_mime_types = ['application/pdf', 'image/jpeg', 'image/png', 'text/plain', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
            
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            for file in files_list:
                if not file.content_type:
                    raise HTTPException(status_code=400, detail=f"File content type could not be determined for {file.filename}.")

                if file.content_type not in allowed_mime_types:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type} for file {file.filename}. Supported types: {', '.join(allowed_mime_types)}")

                print(f"Received file: {file.filename}, type: {file.content_type}")
                file_content = await file.read()
                if not file_content:
                    raise HTTPException(status_code=400, detail=f"Uploaded file {file.filename} is empty.")

                print(f"Read {len(file_content)} bytes from file {file.filename}.")
                file_names.append(file.filename)

                temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

                with open(temp_file_path, "wb") as temp_f:
                    temp_f.write(file_content)

                uploaded_file_response = genai.upload_file(
                    path=temp_file_path,
                    display_name=file.filename or f"upload_{uuid.uuid4()}",
                    mime_type=file.content_type
                )

                os.remove(temp_file_path)

                uploaded_file_uris.append(uploaded_file_response.uri)
                uploaded_file_mime_types.append(uploaded_file_response.mime_type)
                print(f"File {file.filename} uploaded successfully. URI: {uploaded_file_response.uri}")
            
            print(f"Total files processed: {len(uploaded_file_uris)}")
        
        # Choose the first file for the message if any was uploaded
        primary_file_uri = None
        primary_file_mime_type = None
        if uploaded_file_uris:
            primary_file_uri = uploaded_file_uris[0]
            primary_file_mime_type = uploaded_file_mime_types[0]

        user_msg = ChatMessage(
            session_id=session_id,
            role='user',
            content=message,
            file_uri=primary_file_uri,
            file_mime_type=primary_file_mime_type
        )
        db.add(user_msg)

        message_count = db.query(ChatMessage).filter_by(session_id=session_id).count()
        if not session.title and message_count == 0:
            session.title = (message[:50] + '...') if len(message) > 50 else message

        db.commit()
        db.refresh(user_msg)
        print(f"User message saved: ID {user_msg.id}, File URI: {user_msg.file_uri}")

        # Get conversation history from database
        db_messages = db.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
        gemini_history, history_file_uri, history_file_mime = build_gemini_history(db_messages)

        # Always append system prompt to each user query but store only the original message in DB
        modified_prompt = f"{SYSTEM_PROMPT}\n\nUser query: {message}"
        
        # Add context about additional files if there are more than one
        if len(uploaded_file_uris) > 1:
            additional_files_info = f"\n\nNote: The user has uploaded {len(uploaded_file_uris)} documents: "
            additional_files_info += ", ".join([f'"{name}"' for name in file_names])
            additional_files_info += f". You're analyzing all these documents together, and the user's query may refer to any of them."
            modified_prompt += additional_files_info
            
        print("Adding system instructions to user message")

        # Create message content with all files
        message_content = []
        # Always include text prompt
        message_content.append({'text': modified_prompt})
        
        # Add all files
        for i, (file_uri, file_mime_type) in enumerate(zip(uploaded_file_uris, uploaded_file_mime_types)):
            message_content.append({
                'file_data': {
                    'mime_type': file_mime_type,
                    'file_uri': file_uri
                }
            })
            print(f"Added file {i+1} to request with URI: {file_uri}")
        
        # If no files in current message, use the most recent file from history
        if not uploaded_file_uris and history_file_uri and history_file_mime:
            message_content.append({
                'file_data': {
                    'mime_type': history_file_mime,
                    'file_uri': history_file_uri
                }
            })
            print(f"Using previous file from history: {history_file_uri}")

        # Generate the response
        try:
            # Start chat with simplified history structure
            chat = model.start_chat(history=gemini_history or [])
            
            # Send message with all files
            response = chat.send_message(
                {'parts': message_content},
                stream=False,
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "top_k": 40,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            # Extract text from response
            full_response = ""
            if hasattr(response, 'text'):
                full_response = response.text
            elif hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'text'):
                        full_response += part.text
                        
            print(f"Generation complete. Response length: {len(full_response)}")
            
            # Store the response in the database
            assistant_msg = ChatMessage(
                session_id=session_id,
                role='model',
                content=full_response,
                file_uri=None,
                file_mime_type=None
            )
            db.add(assistant_msg)
            db.commit()
            db.refresh(assistant_msg)
            
            # Return the response as JSON
            return {
                "id": assistant_msg.id,
                "content": full_response,
                "role": "model",
                "timestamp": assistant_msg.timestamp.isoformat(),
                "num_files_processed": len(uploaded_file_uris)
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    except Exception as e:
        db.rollback()
        print(f"Error processing message or generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get("/chat/sessions")
async def list_sessions(db: Session = Depends(get_db)):
    """List all chat sessions."""
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()

    result = []
    for session in sessions:
        last_user_message = db.query(ChatMessage).filter(
            ChatMessage.session_id == session.id,
            ChatMessage.role == "user" # Look for last USER message
        ).order_by(ChatMessage.timestamp.desc()).first()

        result.append({
            "id": session.id,
            "title": session.title or "New Chat",
            "created_at": session.created_at,
            "last_message": last_user_message.content if last_user_message else None
        })

    return result



@app.get("/chat/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, db: Session = Depends(get_db)):
    try:
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        messages = db.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
        
        return {
            "session": {
                "id": session.id,
                "title": session.title or "New Chat",
                "created_at": session.created_at
            },
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "file_uri": msg.file_uri,
                    "file_mime_type": msg.file_mime_type
                }
                for msg in messages
            ]
        }
    except Exception as e:
        print(f"Error fetching session history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chat history")


@app.delete("/chat/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    try:
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        db.delete(session)
        db.commit()
        
        return {"success": True, "message": "Chat session deleted successfully"}
    except Exception as e:
        db.rollback()
        print(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail="Error deleting chat session")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini Chatbot API is running"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8090))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")
    print(f"Starting server on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, log_level=log_level, reload=True)