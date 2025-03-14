from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime
from persistence.storage import Storage

class MemoryManager:
    """Manages persistence of user learning state and progress."""
    
    def __init__(self, storage: Storage):
        """
        Initialize the memory manager.
        
        Args:
            storage: Storage backend for persistence
        """
        self.storage = storage
        
    async def save_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save the current state to persistent storage.
        
        Args:
            state: Current state to save
            
        Returns:
            Updated state with save confirmation
        """
        user_id = state.get("user_id")
        topic = state.get("current_topic")
        
        if not user_id or not topic:
            state["error"] = "Missing user_id or topic for state persistence"
            return state
        
        # Add timestamp
        state["last_updated"] = datetime.now().isoformat()
        
        # Track interaction metrics
        if "interaction_metrics" not in state:
            state["interaction_metrics"] = {
                "questions_asked": 0,
                "practice_attempts": 0,
                "time_spent": 0,
                "last_interaction": None
            }
        
        # Update metrics based on request type
        if state.get("request_type") == "question":
            state["interaction_metrics"]["questions_asked"] += 1
        elif state.get("request_type") == "practice_evaluation":
            state["interaction_metrics"]["practice_attempts"] += 1
        
        # Update last interaction time
        state["interaction_metrics"]["last_interaction"] = datetime.now().isoformat()
        
        # Calculate time spent if possible
        if state["interaction_metrics"].get("session_start") is None:
            state["interaction_metrics"]["session_start"] = datetime.now().isoformat()
        
        # Save to storage
        await self.storage.save(
            collection="user_states",
            key=f"{user_id}_{topic}",
            data=state
        )
        
        state["state_saved"] = True
        return state
    
    async def load_state(self, user_id: str, topic: str) -> Optional[Dict[str, Any]]:
        """
        Load state from persistent storage.
        
        Args:
            user_id: User identifier
            topic: Topic identifier
            
        Returns:
            Loaded state or None if not found
        """
        try:
            state = await self.storage.load(
                collection="user_states",
                key=f"{user_id}_{topic}"
            )
            return state
        except Exception:
            return None
    
    async def save_checkpoint_result(self, user_id: str, topic: str, subtopic: str, 
                                    result: Dict[str, Any]) -> bool:
        """
        Save checkpoint assessment results.
        
        Args:
            user_id: User identifier
            topic: Topic identifier
            subtopic: Subtopic identifier
            result: Assessment result data
            
        Returns:
            Success status
        """
        key = f"{user_id}_{topic}_{subtopic}_checkpoint"
        result["timestamp"] = datetime.now().isoformat()
        
        try:
            await self.storage.save(
                collection="checkpoint_results",
                key=key,
                data=result
            )
            return True
        except Exception:
            return False

    async def track_learning_analytics(self, user_id: str, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Track detailed learning analytics.
        
        Args:
            user_id: User identifier
            event_type: Type of learning event (e.g., 'question', 'practice', 'assessment')
            data: Event data
            
        Returns:
            Success status
        """
        try:
            # Create a unique key for this event
            timestamp = datetime.now().isoformat()
            key = f"{user_id}_{event_type}_{timestamp}"
            
            # Add metadata
            event_data = {
                "user_id": user_id,
                "event_type": event_type,
                "timestamp": timestamp,
                **data
            }
            
            # Save to analytics collection
            await self.storage.save(
                collection="learning_analytics",
                key=key,
                data=event_data
            )
            return True
        except Exception as e:
            print(f"Error tracking analytics: {str(e)}")
            return False

    async def save_message(self, user_id: str, topic: str, role: str, content: str) -> bool:
        """
        Save a chat message to the conversation history.
        
        Args:
            user_id: User identifier
            topic: Topic identifier
            role: Message role (user or assistant)
            content: Message content
            
        Returns:
            Success status
        """
        try:
            # Load current state
            state = await self.load_state(user_id, topic)
            if not state:
                state = {
                    "user_id": user_id,
                    "current_topic": topic,
                    "chat_history": []
                }
            
            # Ensure chat_history exists
            if "chat_history" not in state:
                state["chat_history"] = []
            
            # Add message to history
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            state["chat_history"].append(message)
            
            # Save updated state
            await self.save_state(state)
            return True
        except Exception as e:
            print(f"Error saving message: {str(e)}")
            return False

    async def delete_state(self, user_id: str, topic: str) -> bool:
        """
        Delete a user's state.
        
        Args:
            user_id: User identifier
            topic: Topic identifier
            
        Returns:
            Success status
        """
        try:
            await self.storage.delete(
                collection="user_states",
                key=f"{user_id}_{topic}"
            )
            return True
        except Exception as e:
            print(f"Error deleting state: {str(e)}")
            return False

    async def get_all_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session metadata
        """
        try:
            # This implementation depends on your storage backend
            # For a file-based storage, you might need to list files with a pattern
            sessions = await self.storage.list_keys(
                collection="user_states",
                prefix=f"{user_id}_"
            )
            
            result = []
            for session_key in sessions:
                # Extract topic from key
                topic = session_key.split('_', 1)[1] if '_' in session_key else "unknown"
                
                # Load basic metadata
                state = await self.storage.load(
                    collection="user_states",
                    key=session_key
                )
                
                if state:
                    result.append({
                        "session_id": f"{user_id}_{topic}",
                        "topic": topic,
                        "last_updated": state.get("last_updated"),
                        "message_count": len(state.get("chat_history", []))
                    })
            
            return result
        except Exception as e:
            print(f"Error getting sessions: {str(e)}")
            return []