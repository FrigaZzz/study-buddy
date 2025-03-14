#!/usr/bin/env python3
import asyncio
import argparse
import os
import sys
import yaml
from typing import Dict, Any, List
import json
from datetime import datetime
import readline  # For better input handling with history
import rich
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress

# Import components
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

# Initialize rich console for better formatting
console = Console()

class CLI:
    """Command-line interface for the AI Learning Assistant."""
    
    def __init__(self):
        self.orchestrator = None
        self.user_id = None
        self.topic = None
        self.session_id = None
        self.history = []
        self.learning_path = []
        self.current_subtopic_index = 0
        self.test_mode = False
        self.current_state = None
        self.learning_preferences = {
            "learning_style": "visual",
            "difficulty": "beginner",
            "pace": "moderate"
        }
        self.active_session = False
        self.commands = {
            "help": self.show_help,
            "status": self.show_status,
            "next": self.next_topic,
            "prev": self.previous_topic,
            "progress": self.show_progress,
            "preferences": self.update_preferences,
            "review": self.review_topic,
            "quiz": self.start_quiz,
            "practice": self.start_practice,
            "exit": self.exit_session
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    async def initialize_components(self, config_path: str = None):
        """Initialize all components needed for the application."""
        # Load configuration
        if not config_path:
            config_path = os.path.join("config", "config.yaml")
        
        try:
            config = self.load_config(config_path)
        except FileNotFoundError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            console.print("[yellow]Creating default configuration...[/yellow]")
            config = self.create_default_config()
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        
        console.print("[bold blue]Initializing components...[/bold blue]")
        
        # Initialize language model with Ollama
        with Progress() as progress:
            task = progress.add_task("[green]Loading models...", total=5)
            
            llm = OllamaLLM(
                model=config.get("llm", {}).get("model_name", "phi3:latest"),
                temperature=config.get("llm", {}).get("temperature", 0.7),
                base_url=config.get("llm", {}).get("base_url", "http://localhost:11434")
            )
            progress.update(task, advance=1)
            
            # Add a separate model for interactive dialogue with potentially higher temperature
            dialogue_llm = OllamaLLM(
                model=config.get("dialogue_llm", {}).get("model_name", "phi3:3.8b"),
                temperature=config.get("dialogue_llm", {}).get("temperature", 0.8),
                base_url=config.get("dialogue_llm", {}).get("base_url", "http://localhost:11434")
            )
            progress.update(task, advance=1)
            
            # Initialize embeddings with updated Ollama import
            embeddings = OllamaEmbeddings(
                model=config.get("embeddings", {}).get("model_name", "phi3:3.8b"),
                base_url=config.get("embeddings", {}).get("base_url", "http://localhost:11434")
            )
            progress.update(task, advance=1)
            
            # Initialize vector store
            vector_store_path = os.path.join("data", "vector_store")
            os.makedirs(vector_store_path, exist_ok=True)
            
            # Check if vector store exists, if not create empty one
            if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
                console.print("[yellow]Creating new vector store...[/yellow]")
                vector_store = FAISS.from_texts(["Initial document"], embeddings)
                vector_store.save_local(vector_store_path)
            else:
                console.print("[green]Loading existing vector store...[/green]")
                # Add allow_dangerous_deserialization=True to fix the ValueError
                vector_store = FAISS.load_local(
                    vector_store_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
            progress.update(task, advance=1)
            
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
            self.orchestrator = Orchestrator(
                learning_agent=learning_agent,
                assessment_agent=assessment_agent,
                memory_manager=memory_manager,
                config=config
            )
            
            # Store curriculum for navigation
            self.curriculum = config.get("curriculum", [])
            progress.update(task, advance=1)
        
        console.print("[bold green]Components initialized successfully![/bold green]")
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration."""
        return {
            "llm": {
                "model_name": "phi3:latest",
                "temperature": 0.7,
                "base_url": "http://localhost:11434"
            },
            "dialogue_llm": {
                "model_name": "phi3:3.8b",
                "temperature": 0.8,
                "base_url": "http://localhost:11434"
            },
            "embeddings": {
                "model_name": "phi3:3.8b",
                "base_url": "http://localhost:11434"
            },
            "rag": {
                "top_k": 5
            },
            "curriculum": [
                {
                    "topic": "python",
                    "subtopics": ["basics", "functions", "classes", "modules"]
                },
                {
                    "topic": "machine_learning",
                    "subtopics": ["basics", "supervised", "unsupervised", "deep_learning"]
                }
            ],
            "assessment": {
                "passing_score": 70,
                "question_count": 3,
                "question_types": ["multiple_choice", "open_ended"],
                "difficulty": "medium"
            }
        }
    
    async def start_session(self, user_id: str, topic: str):
        """Start a new learning session."""
        self.user_id = user_id
        self.topic = topic
        self.session_id = f"{user_id}_{topic}"
        
        console.print(f"[bold]Starting session for user [cyan]{user_id}[/cyan] on topic [green]{topic}[/green][/bold]")
        
        # Initialize session state
        state = await self.orchestrator.start_session(user_id, topic)
        
        # Set the current state
        self.current_state = state
        
        # Get any existing history
        self.history = await self.orchestrator.get_chat_history(self.session_id)
        
        # Set up learning path based on curriculum
        for curriculum_item in self.curriculum:
            if curriculum_item.get("topic") == topic:
                self.learning_path = curriculum_item.get("subtopics", [])
                break
        
        # Print welcome message
        console.print(Panel.fit(
            f"[bold]Welcome to your learning session on [green]{topic}[/green]![/bold]\n\n"
            "Type your messages to interact with the assistant.\n"
            "Type [bold cyan]'exit'[/bold cyan], [bold cyan]'quit'[/bold cyan], or [bold cyan]'q'[/bold cyan] to end the session.\n"
            "Type [bold cyan]'help'[/bold cyan] for more commands.",
            title="AI Learning Assistant",
            border_style="blue"
        ))
        
        # Print history if available
        if self.history:
            console.print("[bold]Previous conversation:[/bold]")
            for msg in self.history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    console.print(f"\n[bold cyan]You:[/bold cyan] {content}")
                else:
                    console.print(f"\n[bold green]Assistant:[/bold green] {content}")
            console.print("\n" + "-"*50 + "\n")
    
    async def process_command(self, command: str):
        """Process CLI commands with better error handling and state validation."""
        try:
            cmd = command.lower().strip()
            if cmd in self.commands:
                if not self._validate_state_for_command(cmd):
                    console.print("[yellow]Invalid state for this command. Try 'help'[/yellow]")
                    return True
                return await self.commands[cmd]()
            return await self.process_message(command)
        except Exception as e:
            console.print(f"[red]Error processing command: {str(e)}[/red]")
            return True
            
    def _validate_state_for_command(self, command: str) -> bool:
        """Validate current state for command execution."""
        required_state = {
            "next": ["current_topic", "learning_path"],
            "prev": ["current_topic", "learning_path"],
            "progress": ["user_id", "current_topic"],
            "quiz": ["current_topic", "mastery_levels"],
            "practice": ["current_topic", "current_subtopic"]
        }
        
        if command not in required_state:
            return True
            
        return all(self.current_state.get(field) for field in required_state[command])
    
    async def process_message(self, message: str):
        """Process a user message and display the response."""
        if not message.strip():
            return True
        
        # Handle special commands
        if message.lower() in ['exit', 'quit', 'q']:
            console.print("[bold yellow]Ending session. Goodbye![/bold yellow]")
            return False
        
        if message.lower() == 'help':
            self.show_help()
            return True
        
        if message.lower() == 'history':
            self.show_history()
            return True
        
        if message.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            return True
        
        if message.lower() == 'curriculum':
            self.show_curriculum()
            return True
        
        if message.lower().startswith('next'):
            return await self.next_subtopic()
        
        if message.lower().startswith('test'):
            return await self.run_test_mode()
        
        # Save user message to history
        await self.orchestrator.memory_manager.save_message(
            self.user_id, self.topic, "user", message
        )
        
        # Process the message
        with console.status("[bold green]Thinking...[/bold green]"):
            response = await self.orchestrator.process_message(message, self.session_id)
        
        # Save assistant response to history
        await self.orchestrator.memory_manager.save_message(
            self.user_id, self.topic, "assistant", response
        )
        
        # Display the response
        console.print(f"\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(response))
        console.print()
        
        return True
    
    def show_help(self):
        """Show available commands."""
        console.print(Panel.fit(
            "[bold cyan]help[/bold cyan]    - Show this help message\n"
            "[bold cyan]history[/bold cyan] - Show conversation history\n"
            "[bold cyan]clear[/bold cyan]   - Clear the screen\n"
            "[bold cyan]curriculum[/bold cyan] - Show learning curriculum\n"
            "[bold cyan]next[/bold cyan]    - Move to next subtopic\n"
            "[bold cyan]test[/bold cyan]    - Run in test mode with automated questions\n"
            "[bold cyan]exit[/bold cyan]    - End the session (also 'quit' or 'q')\n"
            "[bold cyan]<text>[/bold cyan]  - Any other text is sent to the assistant",
            title="Available Commands",
            border_style="cyan"
        ))
    
    def show_history(self):
        """Show conversation history."""
        console.print("[bold]Conversation history:[/bold]")
        for msg in self.history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            if timestamp:
                time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
                if role == "user":
                    console.print(f"\n[dim][{time_str}][/dim] [bold cyan]You:[/bold cyan] {content}")
                else:
                    console.print(f"\n[dim][{time_str}][/dim] [bold green]Assistant:[/bold green] {content}")
            else:
                if role == "user":
                    console.print(f"\n[bold cyan]You:[/bold cyan] {content}")
                else:
                    console.print(f"\n[bold green]Assistant:[/bold green] {content}")
        console.print()
    
    def show_curriculum(self):
        """Show the learning curriculum."""
        for curriculum_item in self.curriculum:
            if curriculum_item.get("topic") == self.topic:
                subtopics = curriculum_item.get("subtopics", [])
                console.print(Panel.fit(
                    "\n".join([
                        f"{'[bold green]✓[/bold green]' if i < self.current_subtopic_index else '[ ]'} {subtopic}"
                        for i, subtopic in enumerate(subtopics)
                    ]),
                    title=f"Curriculum for {self.topic}",
                    border_style="green"
                ))
                return
        
        console.print("[yellow]No curriculum found for this topic.[/yellow]")
    
    async def next_subtopic(self):
        """Move to the next subtopic in the curriculum."""
        if not self.learning_path or self.current_subtopic_index >= len(self.learning_path):
            console.print("[yellow]You've completed all subtopics in this curriculum.[/yellow]")
            return True
        
        next_subtopic = self.learning_path[self.current_subtopic_index]
        self.current_subtopic_index += 1
        
        console.print(f"[bold]Moving to subtopic: [green]{next_subtopic}[/green][/bold]")
        
        # Update state with new subtopic
        state = await self.orchestrator.memory_manager.load_state(self.user_id, self.topic)
        if state:
            state["current_subtopic"] = next_subtopic
            await self.orchestrator.memory_manager.save_state(state)
        
        # Request learning content for this subtopic
        message = f"Please teach me about {next_subtopic} in {self.topic}"
        return await self.process_message(message)
    
    async def run_test_mode(self):
        """Run in test mode with automated questions and user answers."""
        self.test_mode = True
        console.print("[bold yellow]Entering test mode...[/bold yellow]")
        
        # Predefined test questions for different topics
        test_questions = {
            "python": [
                "What are the basic data types in Python?",
                "How do you define a function in Python?",
                "Explain the difference between a list and a tuple.",
                "What is object-oriented programming in Python?",
                "How do you handle exceptions in Python?"
            ],
            "machine_learning": [
                "What is the difference between supervised and unsupervised learning?",
                "Explain what a decision tree is.",
                "What is overfitting and how can you prevent it?",
                "Explain the concept of gradient descent.",
                "What is the difference between precision and recall?"
            ]
        }
        
        # Get questions for current topic or use default
        questions = test_questions.get(self.topic, [
            "What are the key concepts in this topic?",
            "Can you explain the fundamentals of this subject?",
            "What are some practical applications of this knowledge?",
            "What are the most important things to remember about this topic?",
            "How would you explain this topic to a beginner?"
        ])
        
        # Run through test questions
        score = 0
        total_questions = len(questions)
        
        for i, question in enumerate(questions):
            console.print(f"\n[bold cyan]Test Question {i+1}/{total_questions}:[/bold cyan]")
            console.print(f"[cyan]{question}[/cyan]\n")
            
            # Get user's answer
            console.print("[bold cyan]Your Answer:[/bold cyan] (type your answer and press Enter)")
            user_answer = input()
            
            # Save user's question and answer
            await self.orchestrator.memory_manager.save_message(
                self.user_id, self.topic, "user", question
            )
            await self.orchestrator.memory_manager.save_message(
                self.user_id, self.topic, "user", f"My answer: {user_answer}"
            )
            
            # Process the evaluation
            console.print("\n[bold]Evaluating your answer...[/bold]")
            
            # Create evaluation prompt
            eval_message = f"Evaluate this answer to the question: '{question}'\n\nUser's answer: {user_answer}\n\nProvide feedback and score out of 10."
            
            # Get evaluation from the assistant
            response = await self.orchestrator.process_message(eval_message, self.session_id)
            
            # Display evaluation
            console.print("\n[bold green]Evaluation:[/bold green]")
            console.print(Markdown(response))
            
            # Extract score (this is a simple heuristic, might need improvement)
            try:
                score_text = response.lower()
                if "score:" in score_text:
                    score_part = score_text.split("score:")[1].strip().split()[0]
                    question_score = float(score_part.replace("/10", "").strip())
                    score += question_score
                elif "/10" in score_text:
                    for word in score_text.split():
                        if "/10" in word:
                            question_score = float(word.replace("/10", "").strip())
                            score += question_score
                            break
            except:
                # If we can't extract a score, make a guess based on positive words
                positive_words = ["excellent", "perfect", "great", "good", "correct"]
                question_score = 0
                for word in positive_words:
                    if word in response.lower():
                        question_score += 2
                question_score = min(question_score, 10)
                score += question_score
            
            # Pause between questions
            if i < total_questions - 1:
                console.print("\n[dim]Press Enter for next question...[/dim]")
                input()
        
        # Calculate final score as percentage
        final_score = (score / (total_questions * 10)) * 100
        
        console.print(f"\n[bold green]Test completed![/bold green]")
        console.print(f"[bold]Your score: {final_score:.1f}%[/bold]")
        
        # Provide overall feedback
        if final_score >= 90:
            console.print("[bold green]Excellent! You have mastered this topic.[/bold green]")
        elif final_score >= 75:
            console.print("[bold green]Very good! You have a strong understanding of this topic.[/bold green]")
        elif final_score >= 60:
            console.print("[bold yellow]Good! You understand the basics but might want to review some concepts.[/bold yellow]")
        else:
            console.print("[bold yellow]You might need more practice with this topic. Consider reviewing the material again.[/bold yellow]")
        
        self.test_mode = False
        return True
    
    async def run_interactive(self):
        """Run the CLI in interactive mode."""
        try:
            while True:
                # Current implementation is too simple
                message = input("[bold cyan]You:[/bold cyan] ")
                result = await self.process_command(message)
                if result is False:
                    break
                # Missing:
                # - State validation
                # - Session recovery
                # - Progress tracking
                # - Learning path navigation
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            import traceback
            traceback.print_exc()

    async def navigate_learning_path(self):
        """Navigate through the learning path."""
        if not self.current_state or not self.current_state.get("current_learning_path"):
            # Initialize learning path based on topic and user level
            self.current_state["current_learning_path"] = await self._generate_learning_path()
        
        # Add navigation commands
        navigation_commands = {
            "next": self._move_to_next_topic,
            "prev": self._move_to_previous_topic,
            "jump": self._jump_to_topic,
            "review": self._review_current_topic
        }

    async def recover_session(self):
        """Recover previous session state."""
        try:
            saved_state = await self.orchestrator.memory_manager.load_state(self.user_id, self.topic)
            if saved_state:
                # Validate and restore state
                if self._validate_state(saved_state):
                    self.current_state = saved_state
                    return True
            return False
        except Exception as e:
            console.print(f"[red]Error recovering session: {str(e)}[/red]")
            return False

    async def track_progress(self):
        """Track and display learning progress."""
        if not self.current_state:
            return
        
        progress_data = {
            "completed_topics": [],
            "mastery_levels": {},
            "time_spent": {},
            "exercises_completed": 0,
            "assessment_scores": []
        }
        
        # Update progress visualization
        self._display_progress(progress_data)

    # Add missing command handlers
    async def show_status(self):
        """Show current session status."""
        if not self.current_state:
            console.print("[yellow]No active session.[/yellow]")
            return True
            
        console.print(Panel.fit(
            f"[bold]Current Status[/bold]\n\n"
            f"Topic: [green]{self.current_state.get('current_topic', 'None')}[/green]\n"
            f"Subtopic: [blue]{self.current_state.get('current_subtopic', 'None')}[/blue]\n"
            f"Progress: {self.current_state.get('progress', 0)}%\n"
            f"Learning Style: {self.current_state.get('learning_style', 'Not set')}\n"
            f"Last Activity: {self.current_state.get('last_interaction_time', 'Never')}",
            title="Session Status",
            border_style="cyan"
        ))
        return True

    async def next_topic(self):
        """Move to the next topic in the learning path."""
        if not self.current_state or not self.current_state.get("current_learning_path"):
            console.print("[yellow]No learning path defined.[/yellow]")
            return True
            
        current_path = self.current_state["current_learning_path"]
        current_topic = self.current_state.get("current_topic")
        
        try:
            current_index = current_path.index(current_topic)
            if current_index + 1 < len(current_path):
                next_topic = current_path[current_index + 1]
                await self.start_session(self.user_id, next_topic)
                return True
            else:
                console.print("[yellow]You've reached the end of the learning path![/yellow]")
        except ValueError:
            console.print("[red]Error: Current topic not found in learning path.[/red]")
        return True

    async def previous_topic(self):
        """Move to the previous topic in the learning path."""
        if not self.current_state or not self.current_state.get("current_learning_path"):
            console.print("[yellow]No learning path defined.[/yellow]")
            return True
            
        current_path = self.current_state["current_learning_path"]
        current_topic = self.current_state.get("current_topic")
        
        try:
            current_index = current_path.index(current_topic)
            if current_index > 0:
                prev_topic = current_path[current_index - 1]
                await self.start_session(self.user_id, prev_topic)
                return True
            else:
                console.print("[yellow]You're at the beginning of the learning path![/yellow]")
        except ValueError:
            console.print("[red]Error: Current topic not found in learning path.[/red]")
        return True

    async def show_progress(self):
        """Show learning progress."""
        await self.track_progress()
        return True

    async def update_preferences(self):
        """Update learning preferences."""
        preferences = {
            "learning_style": ["visual", "auditory", "reading/writing", "kinesthetic"],
            "difficulty": ["beginner", "intermediate", "advanced"],
            "pace": ["slow", "moderate", "fast"]
        }
        
        console.print("\n[bold]Update Learning Preferences[/bold]")
        
        new_preferences = {}
        for pref, options in preferences.items():
            console.print(f"\n[cyan]Select {pref}:[/cyan]")
            for i, option in enumerate(options, 1):
                console.print(f"{i}. {option}")
            
            while True:
                try:
                    choice = int(input("Enter number: "))
                    if 1 <= choice <= len(options):
                        new_preferences[pref] = options[choice-1]
                        break
                    else:
                        console.print("[red]Invalid choice. Try again.[/red]")
                except ValueError:
                    console.print("[red]Please enter a number.[/red]")
        
        self.learning_preferences = new_preferences
        if self.current_state:
            self.current_state["user_preferences"] = new_preferences
            await self.orchestrator.memory_manager.save_state(self.current_state)
        
        console.print("\n[green]Preferences updated successfully![/green]")
        return True

    async def review_topic(self):
        """Review current topic."""
        if not self.current_state:
            console.print("[yellow]No active session to review.[/yellow]")
            return True
            
        topic = self.current_state.get("current_topic")
        subtopic = self.current_state.get("current_subtopic")
        
        console.print(f"\n[bold]Reviewing {topic}/{subtopic}[/bold]")
        
        # Get summary of current topic
        with console.status("[bold green]Generating review...[/bold green]"):
            review = await self.orchestrator.process_message(
                f"Please provide a concise review of {subtopic} in {topic}",
                self.session_id
            )
        
        console.print(Markdown(review))
        return True

    async def start_quiz(self):
        """Start a quiz on the current topic."""
        if not self.current_state:
            console.print("[yellow]No active session for quiz.[/yellow]")
            return True
        
        try:
            # Initialize quiz state if needed
            if "mastery_levels" not in self.current_state:
                self.current_state["mastery_levels"] = {}
            
            self.current_state["request_type"] = "assessment"
            await self.process_message("start quiz")
            return True
        except Exception as e:
            console.print(f"[red]Error starting quiz: {str(e)}[/red]")
            return True

    async def start_practice(self):
        """Start practice exercises."""
        if not self.current_state:
            console.print("[yellow]No active session for practice.[/yellow]")
            return True
            
        self.current_state["request_type"] = "practice"
        await self.process_message("generate practice exercises")
        return True

    async def exit_session(self):
        """Exit the current session."""
        if self.current_state:
            await self.orchestrator.memory_manager.save_state(self.current_state)
        console.print("[bold yellow]Ending session. Goodbye![/bold yellow]")
        return False

    def _display_progress(self, progress_data: Dict[str, Any]):
        """Display progress information."""
        console.print(Panel.fit(
            "\n".join([
                "[bold]Learning Progress[/bold]",
                f"Completed Topics: {len(progress_data['completed_topics'])}",
                f"Exercises Completed: {progress_data['exercises_completed']}",
                f"Average Assessment Score: {sum(progress_data['assessment_scores'] or [0])/len(progress_data['assessment_scores']) if progress_data['assessment_scores'] else 0:.1f}%",
                "\n[bold]Mastery Levels[/bold]",
                *[f"{topic}: {level:.1f}%" for topic, level in progress_data['mastery_levels'].items()],
                "\n[bold]Time Spent[/bold]",
                *[f"{topic}: {time:.1f} minutes" for topic, time in progress_data['time_spent'].items()]
            ]),
            title="Progress Report",
            border_style="green"
        ))

async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="AI Learning Assistant CLI")
    parser.add_argument("--user", "-u", default="test_user", help="User ID")
    parser.add_argument("--topic", "-t", default="python", help="Learning topic")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    
    args = parser.parse_args()
    
    cli = CLI()
    await cli.initialize_components(args.config)
    await cli.start_session(args.user, args.topic)
    
    if args.test:
        await cli.run_test_mode()
    
    await cli.run_interactive()

if __name__ == "__main__":
    asyncio.run(main()) 