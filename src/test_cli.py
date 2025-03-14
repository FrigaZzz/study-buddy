import asyncio
from cli import CLI
from rich import console

async def test_learning_flow():
    """Test the learning flow with proper state initialization."""
    # Initialize CLI
    cli = CLI()
    await cli.initialize_components()
    
    # Start session first
    await cli.start_session("test_user", "list")
    
    # Test sequence
    commands = [
        "status",  # Check initial state
        "preferences",  # Set learning preferences
        "help",  # View available commands
        "start",  # Start learning content
        "practice",  # Try practice exercises
        "quiz",  # Take a quiz
        "progress",  # Check progress
        "next",  # Move to next topic
        "review",  # Review material
        "exit"  # End session
    ]
    
    # Process each command
    for cmd in commands:
        try:
            result = await cli.process_command(cmd)
            if result is False:
                break
            
            # Add delay between commands
            await asyncio.sleep(2)
        except Exception as e:
            console.print(f"[red]Error processing command '{cmd}': {str(e)}[/red]")
            break

if __name__ == "__main__":
    asyncio.run(test_learning_flow()) 