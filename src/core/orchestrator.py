from typing import Dict, Any, List, Callable
import langgraph.graph as lg
from agents.learning_agent import LearningAgent
from agents.assessment_agent import AssessmentAgent
from .memory_manager import MemoryManager
import asyncio
from datetime import datetime

class Orchestrator:
    """Orchestrates the flow between different agents in the learning framework."""
    
    def __init__(
        self,
        learning_agent: LearningAgent,
        assessment_agent: AssessmentAgent,
        memory_manager: MemoryManager,
        config: Dict[str, Any] = None
    ):
        self.learning_agent = learning_agent
        self.assessment_agent = assessment_agent
        self.memory_manager = memory_manager
        self.config = config or {}
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the LangGraph workflow."""
        from typing import TypedDict, Optional, List, Annotated
        from typing_extensions import Annotated
        
        # Define a state type for our graph with Annotated fields for multi-value updates
        class State(TypedDict, total=False):
            user_id: str
            current_topic: str
            current_subtopic: str
            progress: int
            checkpoint_results: List[dict]
            last_agent: Optional[str]
            learning_content: Optional[str]
            assessment_questions: Optional[str]
            user_responses: Optional[str]
            assessment_evaluation: Optional[str]
            sources: Optional[List[str]]
            state_saved: Optional[bool]
            error: Optional[str]
            request_type: Optional[str]
            user_question: Optional[str]
            question_response: Optional[str]
            practice_content: Optional[str]
            practice_solution: Optional[str]
            practice_feedback: Optional[str]
            learning_style: Optional[str]
            subtopic_completed: Optional[bool]
            current_learning_path: Optional[List[str]]  # Track progression through topics
            user_preferences: Optional[Dict[str, Any]]  # Store learning preferences
            mastery_levels: Optional[Dict[str, float]]  # Track topic mastery
            last_interaction_time: Optional[str]  # Track engagement timing
            active_exercises: Optional[List[Dict]]  # Track ongoing exercises
        
        # Create the state graph with the state type
        workflow = lg.StateGraph(State)
        
        # Create a wrapper for load_state to handle the topic argument
        async def load_state_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            """Wrapper for load_state that extracts topic from state."""
            user_id = state.get("user_id")
            topic = state.get("current_topic")
            if not user_id or not topic:
                return state
            try:
                loaded_state = await self.memory_manager.load_state(user_id, topic)
                # Only return loaded state if it exists, otherwise keep current state
                if loaded_state:
                    # Preserve request_type from original state
                    request_type = state.get("request_type")
                    if request_type:
                        loaded_state["request_type"] = request_type
                    return loaded_state
                return state
            except Exception as e:
                print(f"Error in load_state_wrapper: {str(e)}")
                return state
        
        # Fix the router function to properly handle state
        def _route_request(state):
            """Route requests based on request_type."""
            request_type = state.get("request_type", "learning_content")
            print(f"Routing request: {request_type}")
            
            if request_type == "question":
                return "question"
            elif request_type == "practice":
                return "practice"
            elif request_type == "practice_evaluation":
                return "practice_evaluation"
            elif request_type == "assessment":
                return "assessment"
            else:
                return "learning_content"
        
        # Add nodes to the graph
        workflow.add_node("learning", self.learning_agent.process)
        workflow.add_node("assessment", self.assessment_agent.process)
        workflow.add_node("evaluate", self.assessment_agent.evaluate_response)
        workflow.add_node("process_evaluation", self._process_evaluation)
        workflow.add_node("save_state", self.memory_manager.save_state)
        workflow.add_node("load_state", load_state_wrapper)
        workflow.add_node("next_content", self._handle_next_content)
        workflow.add_node("router", lambda x: x)  # Identity function
        
        # Add nodes for interactive features
        workflow.add_node("answer_question", self.learning_agent.answer_question)
        workflow.add_node("generate_practice", self.learning_agent.generate_practice)
        workflow.add_node("evaluate_practice", self.learning_agent.evaluate_practice)
        
        # Simplify the graph structure to avoid concurrent updates
        # Start with load_state
        workflow.add_edge("load_state", "router")
        
        # Route based on request type
        workflow.add_conditional_edges(
            "router",
            _route_request,
            {
                "question": "answer_question",
                "practice": "generate_practice",
                "practice_evaluation": "evaluate_practice",
                "assessment": "assessment",
                "learning_content": "next_content"
            }
        )
        
        # Connect interactive nodes to save_state
        workflow.add_edge("answer_question", "save_state")
        workflow.add_edge("generate_practice", "save_state")
        workflow.add_edge("evaluate_practice", "save_state")
        
        # Connect next_content to learning
        workflow.add_edge("next_content", "learning")
        
        # Connect learning to assessment or save based on condition
        workflow.add_conditional_edges(
            "learning",
            self._should_assess,
            {
                "assessment": "assessment",
                "continue": "save_state"
            }
        )
        
        # Connect assessment flow
        workflow.add_edge("assessment", "evaluate")
        workflow.add_edge("evaluate", "process_evaluation")
        workflow.add_edge("process_evaluation", "save_state")
        
        # Set the entry point
        workflow.set_entry_point("load_state")
        
        return workflow.compile()
    
    async def start_session(self, user_id: str, topic: str) -> Dict[str, Any]:
        """
        Start a new learning session.
        
        Args:
            user_id: Identifier for the user
            topic: The main topic to learn about
            
        Returns:
            Initial state for the session
        """
        # Load any existing state for this user and topic
        initial_state = await self.memory_manager.load_state(user_id, topic)
        
        if not initial_state:
            # Create a new state if none exists
            initial_state = {
                "user_id": user_id,
                "current_topic": topic,
                "current_subtopic": self.config.get("curriculum", [])[0]["subtopics"][0],
                "progress": 0,
                "checkpoint_results": [],
                "last_agent": None
            }
        
        return initial_state
    
    async def process_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single step in the learning flow.
        
        Args:
            state: Current state of the learning session
            
        Returns:
            Updated state after processing
        """
        # Execute the graph for one step
        result = await self.graph.ainvoke(state)
        return result

    def _should_assess(self, state):
        """Determine if we should assess the user at this point."""
        # If explicitly requesting assessment, go to assessment
        if state.get("request_type") == "assessment":
            return "assessment"
        
        # If subtopic is marked as completed, we should assess
        if state.get("subtopic_completed") == True:
            # Mark that we're going to assessment
            state["last_agent"] = "learning"
            return "assessment"
        
        # Mark that we're staying with learning
        state["last_agent"] = "learning"
        return "continue"

    def _handle_next_content(self, state):
        """Handle progression to next content when request_type is learning_content."""
        # Only progress if explicitly requested
        if state.get("request_type") == "learning_content":  # Fixed the condition syntax
            # If we're coming from assessment and have passed, move to next subtopic
            if state.get("last_agent") == "assessment" and state.get("assessment_passed") == True:
                self._advance_to_next_subtopic(state)
                # Clear assessment data
                if "assessment_questions" in state:
                    del state["assessment_questions"]
                if "assessment_evaluation" in state:
                    del state["assessment_evaluation"]
                if "user_responses" in state:
                    del state["user_responses"]
                if "assessment_passed" in state:
                    del state["assessment_passed"]
                # Reset subtopic completion flag
                state["subtopic_completed"] = False
            
            # If we've already seen content for this subtopic, mark it as completed
            # This will trigger assessment on the next step
            elif state.get("learning_content") and not state.get("subtopic_completed"):
                # Mark current subtopic as completed to trigger assessment next
                state["subtopic_completed"] = True
            
            # Clear previous learning content to force generating new content
            state["learning_content"] = None
        
        return state

    def _advance_to_next_subtopic(self, state):
        """Advance to the next subtopic or topic in the curriculum."""
        curriculum = self.config.get("curriculum", [])
        current_topic = state.get("current_topic")
        current_subtopic = state.get("current_subtopic")
        
        # Find current position in curriculum and move to next
        for topic_idx, topic_data in enumerate(curriculum):
            if topic_data["topic"] == current_topic:
                subtopics = topic_data.get("subtopics", [])
                for subtopic_idx, subtopic in enumerate(subtopics):
                    if subtopic == current_subtopic:
                        # Found current position, now move to next
                        if subtopic_idx < len(subtopics) - 1:
                            # Move to next subtopic in same topic
                            state["current_subtopic"] = subtopics[subtopic_idx + 1]
                            return
                        elif topic_idx < len(curriculum) - 1:
                            # Move to first subtopic of next topic
                            state["current_topic"] = curriculum[topic_idx + 1]["topic"]
                            state["current_subtopic"] = curriculum[topic_idx + 1]["subtopics"][0]
                            return
                        # If we're at the end of the curriculum, don't change anything
                        return

    def _process_evaluation(self, state):
        """Process assessment evaluation and track results."""
        # If we have assessment results, save them
        if "assessment_evaluation" in state:
            # Extract score from evaluation (assuming it's in the text)
            evaluation_text = state.get("assessment_evaluation", "")
            import re
            score_match = re.search(r'(\d+)%', evaluation_text)
            score = int(score_match.group(1)) if score_match else 0
            
            # Determine if assessment was passed (e.g., score >= 70%)
            passed = score >= 70
            state["assessment_passed"] = passed
            
            # Add to checkpoint results
            checkpoint_result = {
                "topic": state.get("current_topic"),
                "subtopic": state.get("current_subtopic"),
                "score": score,
                "timestamp": datetime.now().isoformat(),
                "passed": passed
            }
            
            if "checkpoint_results" not in state:
                state["checkpoint_results"] = []
            
            state["checkpoint_results"].append(checkpoint_result)
            
            # Track analytics
            try:
                asyncio.create_task(
                    self.memory_manager.track_learning_analytics(
                        state.get("user_id", "unknown"),
                        "assessment",
                        checkpoint_result
                    )
                )
            except Exception as e:
                print(f"Error tracking analytics: {str(e)}")
        
        return state

    async def _process_assessment_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process assessment evaluation."""
        # Get user responses and assessment questions
        user_responses = state.get("user_responses", "")
        assessment_questions = state.get("raw_assessment_questions", state.get("assessment_questions", ""))
        
        if not user_responses or not assessment_questions:
            state["error"] = "Missing user responses or assessment questions"
            return state
        
        # Evaluate the responses
        evaluation = await self.assessment_agent.evaluate_assessment(
            user_responses=user_responses,
            assessment_questions=assessment_questions
        )
        
        # Add evaluation to state
        state["assessment_evaluation"] = evaluation
        
        # Check if the user passed the assessment
        import re
        score_match = re.search(r'(\d+)%', evaluation)
        score = int(score_match.group(1)) if score_match else 0
        passed = score >= self.config.get("assessment", {}).get("passing_score", 70)
        
        # Update state with assessment results
        state["assessment_passed"] = passed
        state["subtopic_completed"] = passed
        
        # Record checkpoint result
        if state.get("current_topic") and state.get("current_subtopic"):
            checkpoint_result = {
                "topic": state.get("current_topic"),
                "subtopic": state.get("current_subtopic"),
                "score": score,
                "timestamp": datetime.now().isoformat(),
                "passed": passed
            }
            
            if "checkpoint_results" not in state:
                state["checkpoint_results"] = []
            
            state["checkpoint_results"].append(checkpoint_result)
            
            # Track analytics - FIX: Use a proper async approach
            try:
                # Create a new event loop for this task if needed
                if asyncio.get_event_loop().is_running():
                    # If we're in a running event loop, create a task
                    asyncio.create_task(
                        self.memory_manager.track_learning_analytics(
                            state.get("user_id", "unknown"),
                            "assessment",
                            checkpoint_result
                        )
                    )
                else:
                    # If no event loop is running, run the coroutine directly
                    await self.memory_manager.track_learning_analytics(
                        state.get("user_id", "unknown"),
                        "assessment",
                        checkpoint_result
                    )
            except Exception as e:
                print(f"Error tracking analytics: {str(e)}")
        
        return state

    async def process_message(self, message: str, session_id: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            message: The user's message
            session_id: Identifier for the user's session
            
        Returns:
            Response text
        """
        # Parse session_id to extract user_id and topic
        parts = session_id.split('_', 1)
        user_id = parts[0]
        topic = parts[1] if len(parts) > 1 else "general"
        
        # Load or create state
        state = await self.memory_manager.load_state(user_id, topic)
        if not state:
            # Initialize new state
            state = {
                "user_id": user_id,
                "current_topic": topic,
                "current_subtopic": self.config.get("curriculum", [])[0]["subtopics"][0] if self.config.get("curriculum") else "introduction",
                "progress": 0,
                "checkpoint_results": [],
                "last_agent": None
            }
        
        # Determine request type based on message content
        if "?" in message or message.lower().startswith(("what", "how", "why", "can", "could")):
            state["request_type"] = "question"
            state["user_question"] = message
        elif message.lower().startswith(("practice", "exercise")):
            state["request_type"] = "practice"
        elif state.get("practice_content") and message.strip():
            state["request_type"] = "practice_evaluation"
            state["practice_solution"] = message
        elif message.lower().startswith(("assess", "test", "quiz", "evaluate")):
            state["request_type"] = "assessment"
        else:
            state["request_type"] = "learning_content"
        
        # Process the message through the graph
        result = await self.graph.ainvoke(state)
        
        # Determine the response based on request type
        if result.get("request_type") == "question":
            response = result.get("question_response", "I couldn't find an answer to your question.")
        elif result.get("request_type") == "practice":
            response = result.get("practice_content", "I couldn't generate practice exercises.")
        elif result.get("request_type") == "practice_evaluation":
            response = result.get("practice_feedback", "I couldn't evaluate your solution.")
        elif result.get("request_type") == "assessment":
            response = result.get("assessment_questions", "I couldn't generate assessment questions.")
        else:
            response = result.get("learning_content", "I couldn't generate learning content.")
        
        return response

    async def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get chat history for a specific session.
        
        Args:
            session_id: Identifier for the user's session
            
        Returns:
            List of chat messages
        """
        # Parse session_id to extract user_id and topic
        parts = session_id.split('_', 1)
        user_id = parts[0]
        topic = parts[1] if len(parts) > 1 else "general"
        
        # Load state
        state = await self.memory_manager.load_state(user_id, topic)
        if not state:
            return []
        
        # Get chat history from state
        history = state.get("chat_history", [])
        return history

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: Identifier for the user's session
            
        Returns:
            Success status
        """
        # Parse session_id to extract user_id and topic
        parts = session_id.split('_', 1)
        user_id = parts[0]
        topic = parts[1] if len(parts) > 1 else "general"
        
        # Delete state
        try:
            await self.memory_manager.delete_state(user_id, topic)
            return True
        except Exception:
            return False