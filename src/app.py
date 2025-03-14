import asyncio
import os
from typing import Dict, Any, List
import yaml
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
# Replace OpenAI imports with Ollama imports
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from core.orchestrator import Orchestrator
from core.memory_manager import MemoryManager
from agents.learning_agent import LearningAgent
from agents.assessment_agent import AssessmentAgent
from persistence.storage import FileStorage
from knowledge_bases.rag_provider import RAGProvider
from knowledge_bases.web_search_provider import WebSearchProvider

# Define Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Create FastAPI app
app = FastAPI(title="AI Learning Assistant API")

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for components
orchestrator = None

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

async def initialize_components():
    """Initialize all components needed for the application."""
    global orchestrator
    
    # Load configuration
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)
    
    # Initialize language model with Ollama instead of OpenAI
    llm = OllamaLLM(
        model=config.get("llm", {}).get("model_name", "phi3:latest"),
        temperature=config.get("llm", {}).get("temperature", 0.7),
        base_url=config.get("llm", {}).get("base_url", "http://localhost:11434")
    )
    
    # Add a separate model for interactive dialogue with potentially higher temperature
    dialogue_llm = OllamaLLM(
        model=config.get("dialogue_llm", {}).get("model_name", "phi3:3.8b"),
        temperature=config.get("dialogue_llm", {}).get("temperature", 0.8),
        base_url=config.get("dialogue_llm", {}).get("base_url", "http://localhost:11434")
    )
    
    # Initialize embeddings with updated Ollama import
    embeddings = OllamaEmbeddings(
        model=config.get("embeddings", {}).get("model_name", "phi3:3.8b"),
        base_url=config.get("embeddings", {}).get("base_url", "http://localhost:11434")
    )
    
    # Initialize vector store
    vector_store_path = os.path.join("data", "vector_store")
    os.makedirs(vector_store_path, exist_ok=True)
    
    # Check if vector store exists, if not create empty one
    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        vector_store = FAISS.from_texts(["Initial document"], embeddings)
        vector_store.save_local(vector_store_path)
    else:
        # Add allow_dangerous_deserialization=True to fix the ValueError
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    # Initialize storage
    storage = FileStorage(base_dir=os.path.join("data", "storage"))
    
    # Initialize memory manager
    memory_manager = MemoryManager(storage=storage)
    
    # Initialize knowledge providers
    rag_provider = RAGProvider(
        vector_store=vector_store,
        embeddings=embeddings,
        config=config.get("rag", {})
    )
    
    web_search_provider = None
    if config.get("web_search", {}).get("api_key"):
        web_search_provider = WebSearchProvider(
            api_key=config.get("web_search", {}).get("api_key"),
            config=config.get("web_search", {})
        )
    
    # Initialize agents
    learning_agent = LearningAgent(
        llm=llm,
        dialogue_llm=dialogue_llm,
        config=config.get("learning_agent", {}),
        rag_provider=rag_provider,
        web_search_provider=web_search_provider
    )
    
    assessment_agent = AssessmentAgent(
        llm=llm,
        config=config.get("assessment_agent", {})
    )
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        learning_agent=learning_agent,
        assessment_agent=assessment_agent,
        memory_manager=memory_manager,
        config=config
    )

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await initialize_components()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request and return a response."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    # Convert the Pydantic model to the format expected by your orchestrator
    # This may need adjustment based on your orchestrator's interface
    formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Process the request through the orchestrator
    # Assuming orchestrator has a process_message method
    response = await orchestrator.process_message(
        message=formatted_messages[-1]["content"],  # Get the latest message
        session_id=request.session_id
    )
    
    return ChatResponse(
        response=response,
        session_id=request.session_id
    )

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a specific session."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    # Assuming orchestrator can retrieve chat history
    history = await orchestrator.get_chat_history(session_id)
    return {"session_id": session_id, "history": history}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    # Assuming orchestrator can delete a session
    success = await orchestrator.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {"status": "success", "message": f"Session {session_id} deleted"}

@app.get("/sessions")
async def get_sessions(user_id: str):
    """Get all sessions for a user."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    sessions = await orchestrator.memory_manager.get_all_sessions(user_id)
    return {"user_id": user_id, "sessions": sessions}

@app.post("/chat/message")
async def save_message(
    user_id: str, 
    topic: str, 
    role: str, 
    content: str = Body(...)
):
    """Save a chat message."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    success = await orchestrator.memory_manager.save_message(user_id, topic, role, content)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save message")
    
    return {"status": "success"}

@app.post("/sessions")
async def create_session(user_id: str, topic: str):
    """Create a new session."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    state = await orchestrator.start_session(user_id, topic)
    return {"session_id": f"{user_id}_{topic}", "state": state}

def start():
    """Start the FastAPI server."""
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()