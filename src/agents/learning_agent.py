from typing import Dict, Any, List, Optional
from langchain.llms.base import BaseLLM
from .base_agent import BaseAgent
from knowledge_bases.rag_provider import RAGProvider
from knowledge_bases.web_search_provider import WebSearchProvider
from datetime import datetime

class LearningAgent(BaseAgent):
    """Agent responsible for generating learning content and answering questions."""
    
    def __init__(
        self,
        llm: BaseLLM,
        dialogue_llm: Optional[BaseLLM] = None,
        config: Dict[str, Any] = None,
        rag_provider: Optional[RAGProvider] = None,
        web_search_provider: Optional[WebSearchProvider] = None
    ):
        """
        Initialize the learning agent.
        
        Args:
            llm: Language model for content generation
            dialogue_llm: Language model optimized for interactive dialogue (optional)
            config: Configuration parameters
            rag_provider: Provider for retrieval-augmented generation
            web_search_provider: Provider for web search capabilities
        """
        super().__init__(llm, config)
        self.dialogue_llm = dialogue_llm or llm
        self.rag_provider = rag_provider
        self.web_search_provider = web_search_provider
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state to generate learning content.
        
        Args:
            state: Current state of the learning session
            
        Returns:
            Updated state with generated learning content
        """
        try:
            # Extract relevant information from state
            topic = state.get("current_topic", "")
            subtopic = state.get("current_subtopic", "")
            depth = state.get("depth", "intermediate")
            learning_style = state.get("learning_style", "mixed")
            use_web_search = state.get("use_web_search", False)
            
            # Debug output
            print(f"LearningAgent processing: {topic}/{subtopic} at {depth} level")
            
            # Prepare context for content generation
            context = []
            sources = []
            
            # Get relevant information from RAG if available
            if self.rag_provider:
                rag_results = await self.rag_provider.retrieve(
                    query=f"{topic} {subtopic}",
                    top_k=5
                )
                if rag_results:
                    context.extend([doc.page_content for doc in rag_results])
                    sources.extend([doc.metadata.get("source", "Unknown") for doc in rag_results])
            
            # Get web search results if enabled
            if use_web_search and self.web_search_provider:
                web_results = await self.web_search_provider.search(
                    query=f"{topic} {subtopic} tutorial",
                    num_results=3
                )
                if web_results:
                    context.extend([result.get("content", "") for result in web_results])
                    sources.extend([result.get("url", "Web search") for result in web_results])
            
            # Prepare prompt for content generation
            prompt = self._create_learning_content_prompt(
                topic=topic,
                subtopic=subtopic,
                context=context,
                depth=depth,
                learning_style=learning_style
            )
            
            # Generate content
            content = await self.llm.ainvoke(prompt)
            
            # Update state with generated content
            state["learning_content"] = content
            state["sources"] = sources
            
            # Mark subtopic as completed after content is shown
            state["subtopic_completed"] = True
            
            return state
        except Exception as e:
            print(f"Error in LearningAgent.process: {str(e)}")
            import traceback
            traceback.print_exc()
            state["error"] = f"Error generating learning content: {str(e)}"
            state["learning_content"] = "Sorry, there was an error generating the learning content. Please try again."
            return state
            
    async def answer_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer a user question about the current topic.
        
        Args:
            state: Current state with user question
            
        Returns:
            Updated state with question response
        """
        question = state.get("user_question", "")
        topic = state.get("current_topic", "")
        subtopic = state.get("current_subtopic", "")
        learning_content = state.get("learning_content", "")
        
        if not question:
            state["question_response"] = "No question provided."
            return state
        
        # Prepare context for answering the question
        context = [learning_content] if learning_content else []
        sources = []
        
        # Get relevant information from RAG if available
        if self.rag_provider:
            rag_results = await self.rag_provider.retrieve(
                query=question,
                top_k=3
            )
            if rag_results:
                context.extend([doc.page_content for doc in rag_results])
                sources.extend([doc.metadata.get("source", "Unknown") for doc in rag_results])
        
        # Get web search results if enabled
        if state.get("use_web_search", False) and self.web_search_provider:
            web_results = await self.web_search_provider.search(
                query=f"{question} {topic} {subtopic}",
                num_results=2
            )
            if web_results:
                context.extend([result.get("content", "") for result in web_results])
                sources.extend([result.get("url", "Web search") for result in web_results])
        
        # Prepare prompt for answering the question
        prompt = self._create_question_prompt(
            question=question,
            context=context,
            topic=topic,
            subtopic=subtopic
        )
        
        # Generate answer
        try:
            # Use dialogue LLM for more conversational responses
            answer = await self.dialogue_llm.ainvoke(prompt)
            
            # Update state with answer
            state["question_response"] = answer
            
            # Track analytics
            try:
                question_data = {
                    "question": question,
                    "topic": topic,
                    "subtopic": subtopic,
                    "timestamp": datetime.now().isoformat()
                }
                # This would be handled by the orchestrator's process_step method
                state["analytics_event"] = {
                    "type": "question",
                    "data": question_data
                }
            except Exception:
                pass
            
            return state
        except Exception as e:
            state["question_response"] = f"Error answering question: {str(e)}"
            return state
            
    async def generate_practice(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate practice exercises for the current topic.
        
        Args:
            state: Current state of the learning session
            
        Returns:
            Updated state with practice exercises
        """
        topic = state.get("current_topic", "")
        subtopic = state.get("current_subtopic", "")
        depth = state.get("depth", "intermediate")
        learning_content = state.get("learning_content", "")
        
        # Prepare prompt for generating practice exercises
        prompt = self._create_practice_prompt(
            topic=topic,
            subtopic=subtopic,
            depth=depth,
            learning_content=learning_content
        )
        
        # Generate practice exercises
        try:
            practice_content = await self.llm.ainvoke(prompt)
            
            # Update state with practice exercises
            state["practice_content"] = practice_content
            
            return state
        except Exception as e:
            state["practice_content"] = f"Error generating practice exercises: {str(e)}"
            return state
            
    async def evaluate_practice(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a user's practice solution.
        
        Args:
            state: Current state with user's practice solution
            
        Returns:
            Updated state with evaluation feedback
        """
        solution = state.get("practice_solution", "")
        practice_content = state.get("practice_content", "")
        topic = state.get("current_topic", "")
        subtopic = state.get("current_subtopic", "")
        
        if not solution or not practice_content:
            state["practice_feedback"] = "No solution or practice exercises provided."
            return state
        
        # Prepare prompt for evaluating the solution
        prompt = self._create_evaluation_prompt(
            solution=solution,
            practice_content=practice_content,
            topic=topic,
            subtopic=subtopic
        )
        
        # Generate evaluation feedback
        try:
            feedback = await self.dialogue_llm.ainvoke(prompt)
            
            # Update state with feedback
            state["practice_feedback"] = feedback
            
            # Track analytics
            try:
                practice_data = {
                    "topic": topic,
                    "subtopic": subtopic,
                    "timestamp": datetime.now().isoformat()
                }
                # This would be handled by the orchestrator's process_step method
                state["analytics_event"] = {
                    "type": "practice",
                    "data": practice_data
                }
            except Exception:
                pass
            
            return state
        except Exception as e:
            state["practice_feedback"] = f"Error evaluating solution: {str(e)}"
            return state
            
    def _create_learning_content_prompt(
        self, 
        topic: str, 
        subtopic: str, 
        context: List[str],
        depth: str = "intermediate",
        learning_style: str = "mixed"
    ) -> str:
        """Create a prompt for generating learning content."""
        # Combine context information
        context_text = "\n\n".join(context) if context else ""
        
        # Adjust instructions based on learning style
        style_instructions = ""
        if learning_style == "visual":
            style_instructions = "Use descriptive language that creates mental images. Include descriptions of diagrams, charts, and visual representations. Use markdown to create visual structure."
        elif learning_style == "auditory":
            style_instructions = "Use conversational language as if explaining verbally. Include analogies and stories. Use a rhythm and flow that would work well if read aloud."
        elif learning_style == "reading/writing":
            style_instructions = "Use clear, well-structured text with definitions, lists, and detailed explanations. Provide references and terminology. Use markdown for clear organization."
        elif learning_style == "kinesthetic":
            style_instructions = "Include practical examples and hands-on activities. Describe how concepts apply to real-world scenarios. Include step-by-step instructions for practical applications."
        else:  # mixed
            style_instructions = "Balance visual descriptions, conversational explanations, well-structured text, and practical examples. Use markdown for organization and clarity."
        
        # Adjust depth of content
        depth_instructions = ""
        if depth == "beginner":
            depth_instructions = "Explain concepts assuming no prior knowledge. Use simple language and focus on fundamentals. Avoid jargon or define it clearly."
        elif depth == "intermediate":
            depth_instructions = "Assume basic familiarity with the topic. Go beyond fundamentals to explore concepts in more detail. Include some technical terminology."
        elif depth == "advanced":
            depth_instructions = "Assume strong background knowledge. Explore complex aspects of the topic in depth. Use technical terminology and advanced concepts."
        
        # Create the full prompt
        prompt = f"""
        You are an expert educator specializing in {topic}.
        
        Create comprehensive learning content about {subtopic} within the topic of {topic}.
        
        {depth_instructions}
        
        {style_instructions}
        
        Format your response using Markdown for better readability. Include:
        1. A brief introduction to the subtopic
        2. Key concepts and principles
        3. Examples and applications
        4. Summary of main points
        
        Reference information:
        {context_text}
        """
        
        return prompt
    
    def _create_question_prompt(
        self, 
        question: str, 
        context: List[str],
        topic: str,
        subtopic: str
    ) -> str:
        """Create a prompt for answering a user question."""
        # Combine context information
        context_text = "\n\n".join(context) if context else ""
        
        # Create the full prompt
        prompt = f"""
        You are an expert educator specializing in {topic}.
        
        Answer the following question about {subtopic} within the topic of {topic}:
        
        Question: {question}
        
        Provide a clear, accurate, and helpful response. If the question is unclear or outside the scope of the topic, politely explain this and provide relevant information about the topic instead.
        
        Format your response using Markdown for better readability.
        
        Reference information:
        {context_text}
        """
        
        return prompt
    
    def _create_practice_prompt(
        self, 
        topic: str, 
        subtopic: str,
        depth: str = "intermediate",
        learning_content: str = ""
    ) -> str:
        """Create a prompt for generating practice exercises."""
        # Adjust difficulty based on depth
        difficulty_instructions = ""
        if depth == "beginner":
            difficulty_instructions = "Create basic exercises that reinforce fundamental concepts. Focus on recognition and recall."
        elif depth == "intermediate":
            difficulty_instructions = "Create moderately challenging exercises that require application of concepts. Include some problem-solving."
        elif depth == "advanced":
            difficulty_instructions = "Create challenging exercises that require deep understanding and complex problem-solving. Include synthesis of multiple concepts."
        
        # Create the full prompt
        prompt = f"""
        You are an expert educator specializing in {topic}.
        
        Create practice exercises about {subtopic} within the topic of {topic}.
        
        {difficulty_instructions}
        
        Include a mix of:
        1. Conceptual questions to test understanding
        2. Application problems to practice skills
        3. At least one open-ended question for deeper thinking
        
        Format your response using Markdown for better readability. Number each exercise clearly.
        
        Reference content:
        {learning_content}
        """
        
        return prompt
    
    def _create_evaluation_prompt(
        self, 
        solution: str, 
        practice_content: str,
        topic: str,
        subtopic: str
    ) -> str:
        """Create a prompt for evaluating a practice solution."""
        # Create the full prompt
        prompt = f"""
        You are an expert educator specializing in {topic}.
        
        Evaluate the following solution to practice exercises about {subtopic} within the topic of {topic}.
        
        Practice exercises:
        {practice_content}
        
        User's solution:
        {solution}
        
        Provide detailed, constructive feedback on the solution. Include:
        1. What was done correctly
        2. Areas for improvement
        3. Explanations for any misconceptions
        4. Suggestions for further practice
        
        Be encouraging and supportive while providing accurate assessment. Format your response using Markdown for better readability.
        """
        
        return prompt

    async def chat(self, message: str, history: List[Dict[str, str]], topic: str = None) -> str:
        """
        Process a direct chat message without going through the orchestrator.
        
        Args:
            message: User message
            history: Chat history
            topic: Optional topic context
            
        Returns:
            Response text
        """
        # Format history for context
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
        
        # Get relevant information from RAG if available
        rag_context = []
        if self.rag_provider and topic:
            rag_results = await self.rag_provider.retrieve(
                query=f"{topic} {message}",
                top_k=3
            )
            if rag_results:
                rag_context = [doc.page_content for doc in rag_results]
        
        # Prepare prompt
        prompt = f"""
        You are an intelligent learning assistant helping with {topic if topic else 'various topics'}.
        
        Recent conversation:
        {context}
        
        User: {message}
        
        Additional context:
        {'\n'.join(rag_context)}
        
        Provide a helpful, accurate, and conversational response.
        """
        
        # Generate response using dialogue LLM for more conversational tone
        try:
            response = await self.dialogue_llm.ainvoke(prompt)
            return response
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}"
