"""
chat.py: Interactive chat functionality for PipeLM
"""
import sys
import time
import requests
import json
from rich.console import Console
from rich.markdown import Markdown
from pipelm.utils import extract_assistant_response

console = Console()

def interactive_chat(base_url: str = "http://localhost:8080", streaming: bool = True) -> None:
    """Run an interactive chat session with the model via the FastAPI server."""
    console.print("[bold blue]PipeLM - Interactive Chat[/bold blue]")
    console.print(f"[bold blue]Streaming: {'ON' if streaming else 'OFF'}[/bold blue]")
    console.print("[dim]Type '/exit' or '/quit' to end the session[/dim]")
    console.print("[dim]Type '/clear' to clear conversation history[/dim]")
    console.print("[dim]Type '/info' to see model information[/dim]")

    messages = [] # context for the conversation

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
                    console.print(f"[green]Model: {health_info.get('model', 'Unknown')}[green]")
                    console.print(f"Status: {health_info.get('status', 'Unknown')}")
                    uptime = health_info.get('uptime', 0)
                    uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
                    console.print(f"[green]Uptime: {uptime_str}[/green]")
                else:
                    console.print(f"[red]Error getting model info: {response.status_code} - {response.text}[/red]")
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Connection error: {e}[/red]")
            continue
        elif not user_input:
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Prepare generation request
        request_data = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": streaming # <-- Pass the streaming flag to the server
        }

        assistant_response = ""
        console.print("[bold magenta]Assistant:[/bold magenta] ", end="") # Print prefix once

        try:
            # Make the API request
            response = requests.post(
                f"{base_url}/generate",
                json=request_data,
                stream=streaming, # <- Crucial: Enable streaming in requests if desired
                timeout=120 # Set a reasonable timeout
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            if streaming:
                # --- Handle Streaming Response ---
                current_line_length = 0
                max_line_length = console.width - 2 # Adjust for potential scrollbars/borders
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        # Simple word wrapping - print word by word if possible
                        words = chunk.split(' ')
                        for i, word in enumerate(words):
                            word_to_print = word + (' ' if i < len(words) - 1 else '')
                            if current_line_length + len(word_to_print) > max_line_length:
                                print() # Move to next line
                                current_line_length = 0

                            print(word_to_print, end="", flush=True) # Print word without newline, flush buffer
                            current_line_length += len(word_to_print)

                        assistant_response += chunk # Append chunk to full response
                print() # Add a final newline after streaming is done

            else:
                # --- Handle Non-Streaming Response ---
                result = response.json()
                generated_text = result.get("generated_text", "")
                # Use extract_assistant_response if needed for non-streaming format
                assistant_response = extract_assistant_response(generated_text)
                # Display the response using Markdown for non-streaming
                console.print(Markdown(assistant_response))
                console.print()

            # Add the *complete* assistant response to history
            if assistant_response:
                 messages.append({"role": "assistant", "content": assistant_response.strip()})
            else:
                # Handle cases where the stream might have been empty or only whitespace
                console.print("[yellow]Assistant produced no output.[/yellow]")
                # Optionally remove the last user message if assistant failed?
                # messages.pop()


        except requests.exceptions.RequestException as e:
            console.print(f"\n[red]Connection error or server error: {e}[/red]")
            console.print("[yellow]Is the server running? Check server logs.[/yellow]")
            # Remove the user message that failed
            if messages and messages[-1]["role"] == "user":
                 messages.pop()
        except Exception as e:
             console.print(f"\n[red]An unexpected error occurred: {e}[/red]")
             if messages and messages[-1]["role"] == "user":
                  messages.pop()


# send_single_message 
def send_single_message(message: str, base_url: str = "http://localhost:8080", image: str = "",stream:str=True, temperature: float = 0.7, max_tokens: int=1024, top_p: float=0.9) -> str:
    """Send a single message to the model and return the response (non-streaming)."""
    messages = [{"role": "user", "content": message}]
    
    if image!="":
        # If an image is provided, add it to the message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": message}
                ]
            },
        ]
        request_data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream, # default to True
            "image": image # Add image if provided
        }
    else:
        request_data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream # default to True
        }
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=request_data
        )
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text", "")
            # Assuming non-streaming still might benefit from extraction
            return extract_assistant_response(generated_text)
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {e}"