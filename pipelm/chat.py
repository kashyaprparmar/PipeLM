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
from typing import List, Dict, Any
console = Console()

def interactive_chat(
    base_url: str = "http://localhost:8080", streaming: bool = True
) -> None:
    """
    Launch an interactive chat session with the local PipeLM server.

    Special commands:
      /exit, /quit -> exit chat
      /clear       -> clear chat history
      /info        -> show server health
    """
    console.print("[bold blue]PipeLM Interactive Chat[/bold blue]")
    console.print(f"[dim]Streaming: {'ON' if streaming else 'OFF'}[/dim]")
    console.print("[dim]Type '/exit' or '/quit' to end[/dim]")
    console.print("[dim]Type '/clear' to clear history[/dim]")
    console.print("[dim]Type '/info' for server health[/dim]\n")

    history: List[Dict[str, Any]] = []
    while True:
        try:
            user_in = console.input("[bold cyan]> [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Goodbye![/red]")
            break

        cmd = user_in.lower()
        if cmd in ("/exit", "/quit"):
            console.print("[red]Exiting chat...[/red]")
            break
        if cmd == "/clear":
            history.clear()
            console.print("[green]History cleared.[/green]")
            continue
        if cmd == "/info":
            try:
                r = requests.get(f"{base_url}/health")
                r.raise_for_status()
                info = r.json()
                console.print(f"[green]Status:[/green] {info['status']}")
                console.print(f"[green]Model:[/green] {info['model']}")
                console.print(f"[green]Uptime:[/green] {int(info['uptime'])}s")
            except Exception as e:
                console.print(f"[red]Health check failed: {e}[/red]")
            continue
        if not user_in:
            continue

        history.append({"role": "user", "content": user_in})
        console.print("[bold magenta]Assistant:[/bold magenta] ", end="")

        try:
            resp = requests.post(
                f"{base_url}/generate",
                json={
                    "messages": history,
                    "stream": streaming,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                stream=streaming,
            )
            resp.raise_for_status()

            if streaming:
                full = ""
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    full += chunk
                print()
                assistant = full
            else:
                data = resp.json()
                assistant = extract_assistant_response(data.get("generated_text", ""))
                console.print(Markdown(assistant))

            history.append({"role": "assistant", "content": assistant.strip()})

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            if history and history[-1]["role"] == "user":
                history.pop()

# send_single_message 
def send_single_message(message: str, base_url: str = "http://localhost:8080", model_type:str = "text2text", image: str = "",stream:str=True, temperature: float = 0.7, max_tokens: int=1024, top_p: float=0.9) -> str:
    """Send a single message to the model and return the response (non-streaming)."""
    messages = [{"role": "user", "content": message}]
    request_data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream # default to True
    }
    if image != "" and model_type == "image2text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "user", "text": message}
                ]
            },
        ]
        request_data["messages"] = messages
        request_data["image"] = image    
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