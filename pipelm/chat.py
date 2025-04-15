"""
chat.py: Interactive chat functionality for PipeLM
"""
import sys
import time
import requests
from rich.console import Console
from rich.markdown import Markdown

from pipelm.utils import extract_assistant_response, simulated_stream

console = Console()

def interactive_chat(base_url: str = "http://localhost:8080") -> None:
    """Run an interactive chat session with the model via the FastAPI server."""
    console.print("[bold blue]PipeLM - Interactive Chat[/bold blue]")
    console.print("[dim]Type '/exit' or '/quit' to end the session[/dim]")
    console.print("[dim]Type '/clear' to clear conversation history[/dim]")
    console.print("[dim]Type '/info' to see model information[/dim]")
    
    messages = []  # context for the conversation
    
    while True:
        try:
            user_input = console.input("[bold cyan]>>> [/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting chat...[/bold red]")
            break
            
        user_input = user_input.strip()
        
        # Handle commands
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("[bold red]Exiting chat...[/bold red]")
            break
        elif user_input.lower() == "/clear":
            messages = []
            console.print("[green]Conversation history cleared.[/green]")
            continue
        elif user_input.lower() == "/info":
            try:
                response = requests.get(f"{base_url}/health")
                if response.status_code == 200:
                    health_info = response.json()
                    console.print(f"[green]Model: {health_info.get('model', 'Unknown')}")
                    console.print(f"Status: {health_info.get('status', 'Unknown')}")
                    uptime = health_info.get('uptime', 0)
                    uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
                    console.print(f"Uptime: {uptime_str}[/green]")
                else:
                    console.print(f"[red]Error getting model info: {response.status_code}[/red]")
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Connection error: {e}[/red]")
            continue
        elif not user_input:
            continue  # Skip empty input
            
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Prepare generation request
        request_data = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # Make the API request
        try:
            console.print("")  # Add some spacing
            simulated_stream("Loading...")
            
            response = requests.post(
                f"{base_url}/generate",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("generated_text", "")
                
                # Extract just the assistant's response from the full generated text
                assistant_response = extract_assistant_response(generated_text)
                
                # Display the response as markdown
                console.print(Markdown(assistant_response))
                console.print("")  # Add some spacing
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": assistant_response})
            else:
                console.print(f"[red]Error: {response.status_code} - {response.text}[/red]")
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Connection error: {e}[/red]")
            console.print("[yellow]Is the server running?[/yellow]")

def send_single_message(message: str, base_url: str = "http://localhost:8080") -> str:
    """Send a single message to the model and return the response."""
    messages = [{"role": "user", "content": message}]
    
    request_data = {
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text", "")
            return extract_assistant_response(generated_text)
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {e}"