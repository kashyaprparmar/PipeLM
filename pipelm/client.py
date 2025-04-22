"""
client.py
Test client for PipeLM generate endpoint
"""

import argparse
import requests
import sys
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def call_generate(base_url: str, messages: list, max_tokens: int, temperature: float, top_p: float, stream: bool):
    """
    Send a generation request to the server, either streaming or non-streaming.
    """
    url = f"{base_url}/generate"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream
    }

    try:
        if stream:
            resp = requests.post(url, json=payload, stream=True)
            resp.raise_for_status()
            console.print("[bold green]Streaming response:[/bold green]")
            full_text = ""
            for chunk in resp.iter_content(chunk_size=None):
                text = chunk.decode(errors="ignore")
                console.print(text, end="", soft_wrap=True)
                full_text += text
            console.print()  # newline
            return full_text.strip()
        else:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("generated_text", "")
            console.print("[bold green]Full response:[/bold green]")
            console.print(Markdown(text))
            return text.strip()
    except requests.RequestException as e:
        console.print(f"[red]Request error:[/red] {e}")
        sys.exit(1)


def check_health(base_url: str):
    """
    Check server health (and implicitly GPU availability on the server).
    """
    try:
        resp = requests.get(f"{base_url}/health")
        resp.raise_for_status()
        info = resp.json()
        console.print(f"[blue]Server status:[/blue] {info.get('status')}\n[blue]Model:[/blue] {info.get('model')}\n[blue]Uptime (s):[/blue] {info.get('uptime'):.2f}")
    except requests.RequestException as e:
        console.print(f"[red]Health check failed:[/red] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test client for PipeLM generate endpoint"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the PipeLM server"
    )
    parser.add_argument(
        "--prompt",
        default="Hello, how are you?",
        help="Prompt text to send to the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (default is streaming ON)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="(Informational) Indicate the server should be running with GPU support"
    )

    args = parser.parse_args()
    stream = not args.no_stream

    console.print(f"[yellow]Checking server health at {args.base_url}...[/yellow]")
    check_health(args.base_url)

    # SETTING STREAM FOR SERVER to FALSE
    stream = False
    
    console.print(f"[yellow]Sending prompt: '{args.prompt}' (streaming={stream})[/yellow]\n")
    messages = [{"role": "user", "content": args.prompt}]

    output = call_generate(
        base_url=args.base_url,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=stream
    )

    console.print(f"\n[green]Final model output:[/green] \n{output}\n")


if __name__ == "__main__":
    main()
