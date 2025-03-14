from typing import Dict, Any, List, Optional, Callable
import asyncio
import gradio as gr
from core.orchestrator import Orchestrator

class WebUI:
    """Web-based user interface for the learning framework."""
    
    def __init__(self, orchestrator: Orchestrator, config: Dict[str, Any] = None):
        """
        Initialize the web UI.
        
        Args:
            orchestrator: Orchestrator for the learning framework
            config: Configuration parameters
        """
        self.orchestrator = orchestrator
        self.config = config or {}
        self.current_state = {}
        self.user_id = None
        
    def build_ui(self):
        """Build the Gradio interface."""
        with gr.Blocks(title="Learning Framework") as interface:
            gr.Markdown("# Adaptive Learning Framework")
            
            # Add a status area at the top for processing indicators and messages
            with gr.Row():
                status_area = gr.Markdown("Ready to start learning session", elem_id="status-area")
            
            with gr.Row():
                with gr.Column(scale=1):
                    user_id_input = gr.Textbox(label="User ID")
                    topic_input = gr.Textbox(label="Learning Topic")
                    start_btn = gr.Button("Start Learning Session", variant="primary")
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        use_web_search = gr.Checkbox(label="Use Web Search", value=False)
                        depth_select = gr.Dropdown(
                            label="Learning Depth",
                            choices=["beginner", "intermediate", "advanced"],
                            value="intermediate"
                        )
                        learning_style = gr.Dropdown(
                            label="Learning Style",
                            choices=["visual", "auditory", "reading/writing", "kinesthetic", "mixed"],
                            value="mixed"
                        )
                
                with gr.Column(scale=2):
                    with gr.Tabs() as tabs:
                        with gr.TabItem("Learning Content", id="learning-tab"):
                            content_display = gr.Markdown()
                            
                            # Add interactive elements
                            with gr.Row():
                                question_input = gr.Textbox(
                                    label="Ask a question about this topic",
                                    placeholder="Type your question here...",
                                    lines=2
                                )
                                ask_btn = gr.Button("Ask", variant="secondary")
                            
                            response_display = gr.Markdown()
                            
                            with gr.Row():
                                next_btn = gr.Button("Next", variant="primary")
                                practice_btn = gr.Button("Practice This Concept", variant="secondary")
                        
                        with gr.TabItem("Assessment", id="assessment-tab"):
                            # Add a container for assessment instructions
                            assessment_instructions = gr.Markdown("## Assessment\nComplete the questions below to test your knowledge.")
                            
                            # Replace Box with Column for assessment container
                            with gr.Column() as assessment_container:
                                questions_display = gr.Markdown()
                            
                            # Replace Box with Column for answer container
                            with gr.Column() as answer_container:
                                gr.Markdown("### Your Answers")
                                gr.Markdown("Enter your answers below. For multiple choice questions, enter the letter/number of your choice. For open-ended questions, provide a complete answer.")
                                answers_input = gr.Textbox(
                                    label="Your Answers",
                                    placeholder="1. a\n2. Your answer to question 2...\n3. c",
                                    lines=8
                                )
                            
                            submit_answers_btn = gr.Button("Submit Assessment", variant="primary")
                            
                            # Replace Box with Column for results container
                            with gr.Column(visible=False) as results_container:
                                gr.Markdown("### Assessment Results")
                                feedback_display = gr.Markdown()
                                
                                # Add a button to continue after seeing results
                                continue_btn = gr.Button("Continue to Next Section", variant="primary")
                        
                        with gr.TabItem("Progress", id="progress-tab"):
                            progress_display = gr.Markdown()
                            
                        with gr.TabItem("Interactive Practice", id="practice-tab"):
                            practice_area = gr.Markdown()
                            practice_input = gr.Textbox(
                                label="Your Solution",
                                placeholder="Enter your solution here...",
                                lines=5
                            )
                            check_practice_btn = gr.Button("Check Solution", variant="primary")
                            practice_feedback = gr.Markdown()
            
            # Event handlers with loading indicators
            start_btn.click(
                fn=lambda: "Processing... Starting learning session",
                inputs=None,
                outputs=status_area,
                queue=False
            ).then(
                fn=self._handle_start_session,
                inputs=[user_id_input, topic_input, use_web_search, depth_select, learning_style],
                outputs=[content_display, progress_display, status_area]
            )
            
            next_btn.click(
                fn=lambda: "Processing... Loading next content",
                inputs=None,
                outputs=status_area,
                queue=False
            ).then(
                fn=self._handle_next,
                inputs=[],
                outputs=[content_display, questions_display, progress_display, status_area]
            )
            
            submit_answers_btn.click(
                fn=lambda: "Processing... Evaluating answers",
                inputs=None,
                outputs=status_area,
                queue=False
            ).then(
                fn=self._handle_submit_answers,
                inputs=[answers_input],
                outputs=[feedback_display, progress_display, status_area]
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=results_container
            )
            
            # Add new event handlers for interactive elements
            ask_btn.click(
                fn=lambda: "Processing... Answering question",
                inputs=None,
                outputs=status_area,
                queue=False
            ).then(
                fn=self._handle_question,
                inputs=[question_input],
                outputs=[response_display, status_area]
            )
            
            practice_btn.click(
                fn=lambda: "Processing... Generating practice exercises",
                inputs=None,
                outputs=status_area,
                queue=False
            ).then(
                fn=self._handle_practice_request,
                inputs=[],
                outputs=[practice_area, status_area]
            )
            
            check_practice_btn.click(
                fn=lambda: "Processing... Evaluating solution",
                inputs=None,
                outputs=status_area,
                queue=False
            ).then(
                fn=self._handle_check_practice,
                inputs=[practice_input],
                outputs=[practice_feedback, status_area]
            )
            
            # Add custom CSS and JavaScript without using _js parameter
            interface.load(
                fn=lambda: None,
                inputs=None,
                outputs=None
            )

            # Add JavaScript for tab switching using a separate HTML component
            with gr.Row(visible=False):  # Hidden component for JS injection
                js_html = gr.HTML("""
                <script>
                    // Add function to switch tabs programmatically
                    function switchToTab(tabName) {
                        // Find all tab buttons
                        const tabButtons = document.querySelectorAll('.tabs button');
                        // Find the button with the matching text
                        for (let button of tabButtons) {
                            if (button.textContent.includes(tabName)) {
                                button.click();
                                break;
                            }
                        }
                    }
                    
                    // Add custom CSS
                    const style = document.createElement('style');
                    style.textContent = `
                        #status-area {
                            background-color: #f0f0f0;
                            padding: 10px;
                            border-radius: 5px;
                            margin-bottom: 15px;
                            font-weight: bold;
                            text-align: center;
                        }
                        
                        .tabs {
                            margin-top: 10px;
                        }
                        
                        button[variant="primary"] {
                            background-color: #2e7d32;
                        }
                        
                        button[variant="secondary"] {
                            background-color: #1976d2;
                        }
                    `;
                    document.head.appendChild(style);
                </script>
                """)

            # Add a separate button to switch tabs after practice is generated
            with gr.Row(visible=False):
                switch_tab_btn = gr.Button("Switch to Practice Tab")
                switch_tab_btn.click(
                    fn=lambda: None,
                    inputs=None,
                    outputs=None,
                    js="switchToTab('Interactive Practice')"
                )
            
            # Also add a handler for the continue button
            continue_btn.click(
                fn=lambda: "Processing... Loading next content",
                inputs=None,
                outputs=status_area,
                queue=False
            ).then(
                fn=self._handle_next,
                inputs=[],
                outputs=[content_display, questions_display, progress_display, status_area]
            )
            
        return interface
    
    async def _handle_start_session(self, user_id: str, topic: str, use_web_search: bool, depth: str, learning_style: str):
        """Handle starting a new learning session."""
        if not user_id or not topic:
            return "Please provide both User ID and Learning Topic.", "", "Error: Missing required fields"
        
        try:
            # Create initial state
            initial_state = {
                "user_id": user_id,
                "current_topic": topic,
                "current_subtopic": None,  # Will be determined by the learning agent
                "progress": 0,
                "checkpoint_results": [],
                "use_web_search": use_web_search,
                "depth": depth,
                "learning_style": learning_style  # Add learning style preference
            }
            
            # Process initial state
            self.current_state = await self.orchestrator.process_step(initial_state)
            
            # Get content and progress
            content = self.current_state.get("learning_content", "No content available for this topic.")
            progress = self._format_progress()
            
            return content, progress, "Learning session started successfully"
        except Exception as e:
            print(f"Error in _handle_start_session: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error starting session: {str(e)}", "", f"Error: {str(e)}"
    
    async def _handle_next(self):
        """Handle moving to the next learning step."""
        if not self.current_state:
            return "Please start a session first", "", "", "Error: No active session"
        
        try:
            # Debug output
            print(f"Current state before next: {self.current_state}")
            
            # Set request type for next content
            self.current_state["request_type"] = "learning_content"
            
            # Process next step
            self.current_state = await self.orchestrator.process_step(self.current_state)
            
            # Debug output
            print(f"State after next: {self.current_state}")
            
            # Check if we have assessment questions
            if self.current_state.get("assessment_questions"):
                # We're in assessment mode, show the questions
                # Format the questions to hide the correct answers
                raw_questions = self.current_state.get("assessment_questions", "")
                
                # Parse the questions to remove correct answers and explanations
                import json
                try:
                    questions_data = json.loads(raw_questions)
                    formatted_questions = "## Assessment Questions\n\n"
                    
                    for i, q in enumerate(questions_data):
                        q_num = i + 1
                        q_type = q.get("question_type", "")
                        q_text = q.get("question_text", "")
                        
                        formatted_questions += f"**Question {q_num}**: {q_text}\n\n"
                        
                        if q_type == "multiple_choice":
                            options = q.get("options", [])
                            for j, option in enumerate(options):
                                option_letter = chr(97 + j)  # a, b, c, d...
                                formatted_questions += f"{option_letter}) {option}\n"
                            formatted_questions += "\n"
                        
                    questions = formatted_questions
                    
                    # Store the original questions for evaluation
                    self.current_state["raw_assessment_questions"] = raw_questions
                    
                except json.JSONDecodeError:
                    # If not valid JSON, just use as is but try to filter out answers
                    import re
                    questions = re.sub(r'"correct_answer":\s*"[^"]*",?\s*', '', raw_questions)
                    questions = re.sub(r'"explanation":\s*"[^"]*",?\s*', '', questions)
                    
                content = "## Assessment Time\n\nPlease complete the following assessment to continue to the next section:\n\n"
                content += questions
            else:
                # We're in learning mode, show the content
                content = self.current_state.get("learning_content", "No content available for this topic.")
                questions = ""
            
            progress = self._format_progress()
            
            # Make the assessment results container invisible when showing new questions
            return content, questions, progress, "Content loaded successfully"
        except Exception as e:
            print(f"Error in _handle_next: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing next step: {str(e)}", "", "", f"Error: {str(e)}"
    
    async def _handle_submit_answers(self, answers: str):
        """Handle submission of assessment answers."""
        if not self.current_state:
            return "Please start a session first", "", "Error: No active session"
        
        if not answers:
            return "Please provide answers to the assessment questions", "", "Error: No answers provided"
        
        try:
            # Add user responses to state
            self.current_state["user_responses"] = answers
            self.current_state["request_type"] = "assessment_evaluation"
            
            # Process evaluation
            self.current_state = await self.orchestrator.process_step(self.current_state)
            
            # Format the feedback in a more user-friendly way
            raw_feedback = self.current_state.get("assessment_evaluation", "")
            
            # Try to extract the score and format a nicer feedback display
            import re
            score_match = re.search(r'(\d+)%', raw_feedback)
            score = int(score_match.group(1)) if score_match else 0
            
            formatted_feedback = f"## Assessment Results\n\n"
            formatted_feedback += f"### Score: {score}%\n\n"
            
            # Add a visual indicator of pass/fail
            if self.current_state.get("assessment_passed"):
                formatted_feedback += "✅ **Passed!** Congratulations!\n\n"
            else:
                formatted_feedback += "❌ **Needs Improvement**\n\n"
            
            # Add the detailed feedback
            formatted_feedback += "### Feedback\n\n"
            formatted_feedback += raw_feedback
            
            # Add guidance based on assessment result
            if self.current_state.get("assessment_passed"):
                formatted_feedback += "\n\n**Click 'Continue to Next Section' to proceed.**"
            else:
                formatted_feedback += "\n\n**You may want to review this section before continuing. Click 'Continue to Next Section' to try again or continue anyway.**"
            
            progress = self._format_progress()
            
            # Return the feedback and update the status
            return formatted_feedback, progress, "Answers submitted successfully"
        except Exception as e:
            print(f"Error in _handle_submit_answers: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing answers: {str(e)}", "", f"Error: {str(e)}"
    
    def _format_progress(self) -> str:
        """Format progress information for display."""
        if not self.current_state:
            return ""
        
        curriculum = self.config.get("curriculum", [])
        current_topic = self.current_state.get("current_topic")
        current_subtopic = self.current_state.get("current_subtopic")
        checkpoint_results = self.current_state.get("checkpoint_results", [])
        
        # Calculate progress percentage
        total_subtopics = sum(len(topic_data.get("subtopics", [])) for topic_data in curriculum)
        completed_subtopics = 0
        current_found = False
        
        for topic_data in curriculum:
            for subtopic in topic_data.get("subtopics", []):
                if topic_data["topic"] == current_topic and subtopic == current_subtopic:
                    current_found = True
                    break
                completed_subtopics += 1
            if current_found:
                break
        
        progress_pct = int((completed_subtopics / total_subtopics) * 100) if total_subtopics > 0 else 0
        
        # Format checkpoint results
        results_md = "## Checkpoint Results\n\n"
        if checkpoint_results:
            for result in checkpoint_results:
                topic = result.get("topic", "Unknown")
                subtopic = result.get("subtopic", "Unknown")
                score = result.get("score", 0)
                timestamp = result.get("timestamp", "Unknown")
                
                results_md += f"- **{topic}: {subtopic}** - Score: {score}% (Completed: {timestamp})\n"
        else:
            results_md += "No checkpoint results yet.\n"
        
        # Format current progress
        progress_md = f"## Learning Progress: {progress_pct}%\n\n"
        progress_md += f"Current Topic: **{current_topic}**\n\n"
        progress_md += f"Current Subtopic: **{current_subtopic}**\n\n"
        
        # Format curriculum overview
        curriculum_md = "## Curriculum Overview\n\n"
        for topic_data in curriculum:
            topic = topic_data["topic"]
            subtopics = topic_data.get("subtopics", [])
            
            if topic == current_topic:
                curriculum_md += f"### 📚 {topic} (Current)\n\n"
            else:
                curriculum_md += f"### {topic}\n\n"
            
            for subtopic in subtopics:
                if topic == current_topic and subtopic == current_subtopic:
                    curriculum_md += f"- 🔍 **{subtopic}** (Current)\n"
                else:
                    # Check if this subtopic has been completed
                    completed = False
                    for result in checkpoint_results:
                        if result.get("topic") == topic and result.get("subtopic") == subtopic:
                            score = result.get("score", 0)
                            curriculum_md += f"- ✅ {subtopic} (Score: {score}%)\n"
                            completed = True
                            break
                    
                    if not completed:
                        curriculum_md += f"- {subtopic}\n"
            
            curriculum_md += "\n"
        
        # Combine all sections
        return f"{progress_md}\n{curriculum_md}\n{results_md}"
    
    async def _handle_question(self, question: str):
        """Handle user questions about the current topic."""
        if not self.current_state:
            return "Please start a session first", "Error: No active session"
        
        if not question:
            return "Please enter a question", "Error: No question provided"
        
        try:
            # Set request type and question in state
            self.current_state["request_type"] = "question"
            self.current_state["user_question"] = question
            
            # Process the question
            self.current_state = await self.orchestrator.process_step(self.current_state)
            
            # Get the response
            response = self.current_state.get("question_response", "Sorry, I couldn't find an answer to your question.")
            
            return response, "Question answered successfully"
        except Exception as e:
            print(f"Error in _handle_question: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing question: {str(e)}", f"Error: {str(e)}"
    
    async def _handle_practice_request(self):
        """Handle request for practice exercises."""
        if not self.current_state:
            return "Please start a session first", "Error: No active session"
        
        try:
            # Set request type for practice
            self.current_state["request_type"] = "practice"
            
            # Process practice request
            self.current_state = await self.orchestrator.process_step(self.current_state)
            
            # Get practice content
            practice_content = self.current_state.get("practice_content", "No practice exercises available for this topic.")
            
            # Switch to the practice tab
            # Note: This requires additional JS to work properly
            
            return practice_content, "Practice exercises generated successfully"
        except Exception as e:
            print(f"Error in _handle_practice_request: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error generating practice exercises: {str(e)}", f"Error: {str(e)}"
    
    async def _handle_check_practice(self, solution: str):
        """Handle checking practice solutions."""
        if not self.current_state:
            return "Please start a session first", "Error: No active session"
        
        if not solution:
            return "Please enter your solution", "Error: No solution provided"
        
        try:
            # Set request type and solution in state
            self.current_state["request_type"] = "practice_evaluation"
            self.current_state["practice_solution"] = solution
            
            # Process the solution
            self.current_state = await self.orchestrator.process_step(self.current_state)
            
            # Get the feedback
            feedback = self.current_state.get("practice_feedback", "Sorry, I couldn't evaluate your solution.")
            
            return feedback, "Solution evaluated successfully"
        except Exception as e:
            print(f"Error in _handle_check_practice: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error evaluating solution: {str(e)}", f"Error: {str(e)}"
    
    def launch(self, share: bool = False):
        """Launch the web UI."""
        interface = self.build_ui()
        interface.launch(share=share)