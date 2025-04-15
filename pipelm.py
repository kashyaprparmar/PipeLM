#!/usr/bin/env python3
"""
PipeLM: Terminal-based chatting with Hugging Face models.
"""
import os
import sys
import argparse
import getpass
import time
import subprocess
import shutil
import re
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, DownloadColumn
from rich.live import Live
from rich.table import Table
from dotenv import load_dotenv
load_dotenv()

console = Console()

# File download tracking globals
download_stats = {
    "total_files": 0,
    "completed_files": 0,
    "current_file": "",
    "current_size": 0,
    "total_size": 0,
    "start_time": 0,
    "downloaded_bytes": 0,
    "download_speed": 0,  # bytes per second
    "speed_history": [],  # List to track download speeds for smoothing
}

def sanitize_model_name(model_name: str) -> str:
    """Convert model name to a valid directory name."""
    # Replace slashes with underscores and remove special characters
    sanitized = re.sub(r'[^\w\-]', '_', model_name)
    return sanitized

def format_speed(speed_bps: float) -> str:
    """Format speed in bytes per second to a human-readable format."""
    if speed_bps < 1024:
        return f"{speed_bps:.2f} B/s"
    elif speed_bps < 1024 * 1024:
        return f"{speed_bps / 1024:.2f} KB/s"
    elif speed_bps < 1024 * 1024 * 1024:
        return f"{speed_bps / (1024 * 1024):.2f} MB/s"
    else:
        return f"{speed_bps / (1024 * 1024 * 1024):.2f} GB/s"

def huggingface_download_callback(download_info: Dict[str, Any]) -> None:
    """Callback function for tracking huggingface_hub download progress."""
    global download_stats
    
    # Initialize time if this is the first callback
    if download_stats["start_time"] == 0:
        download_stats["start_time"] = time.time()
    
    # Update stats based on new information
    if download_info["status"] == "ongoing":
        download_stats["current_file"] = os.path.basename(download_info["filename"])
        
        if "downloaded" in download_info and "total" in download_info:
            if download_info["total"] > 0:  # Avoid division by zero
                # Update current file progress
                download_stats["current_size"] = download_info["total"]
                download_stats["downloaded_bytes"] = download_info["downloaded"]
                
                # Calculate download speed using a moving average
                elapsed = time.time() - download_stats["start_time"]
                if elapsed > 0:  # Avoid division by zero
                    current_speed = download_info["downloaded"] / elapsed
                    download_stats["speed_history"].append(current_speed)
                    # Keep only the last 5 speed measurements for the moving average
                    if len(download_stats["speed_history"]) > 5:
                        download_stats["speed_history"].pop(0)
                    # Calculate the average speed
                    download_stats["download_speed"] = sum(download_stats["speed_history"]) / len(download_stats["speed_history"])
    
    elif download_info["status"] == "complete":
        # File download complete
        download_stats["completed_files"] += 1
        download_stats["current_file"] = ""
        download_stats["current_size"] = 0
        download_stats["downloaded_bytes"] = 0

def update_download_display() -> None:
    """Update the download progress display in a separate thread."""
    global download_stats
    
    with Live(auto_refresh=False) as live:
        while download_stats["completed_files"] < download_stats["total_files"]:
            # Create a rich table for display
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column()
            table.add_column()
            
            # Progress information
            elapsed = time.time() - download_stats["start_time"] if download_stats["start_time"] > 0 else 0
            progress_pct = (download_stats["completed_files"] / download_stats["total_files"]) * 100 if download_stats["total_files"] > 0 else 0
            
            # Add overall progress
            progress_text = f"Fetching {download_stats['total_files']} files: {progress_pct:.0f}%|"
            bar_length = 40
            completed_bars = int((progress_pct / 100) * bar_length)
            progress_bar = "#" * completed_bars + "-" * (bar_length - completed_bars)
            progress_text += progress_bar + f"| {download_stats['completed_files']}/{download_stats['total_files']}"
            
            if elapsed > 0:
                eta = (elapsed / download_stats["completed_files"]) * (download_stats["total_files"] - download_stats["completed_files"]) if download_stats["completed_files"] > 0 else 0
                time_info = f" [{time.strftime('%M:%S', time.gmtime(elapsed))}<{time.strftime('%M:%S', time.gmtime(eta))}]"
                speed_info = f" {format_speed(download_stats['download_speed'])}"
                progress_text += time_info + speed_info
            
            table.add_row(progress_text, "")
            
            # Add current file information if available
            if download_stats["current_file"]:
                file_text = f"Downloading: {download_stats['current_file']}"
                if download_stats["current_size"] > 0:
                    file_progress = (download_stats["downloaded_bytes"] / download_stats["current_size"]) * 100
                    file_text += f" ({file_progress:.1f}%)"
                table.add_row(file_text, "")
            
            # Update the live display
            live.update(table, refresh=True)
            time.sleep(0.1)

def ensure_model_available(model_name: str, base_model_dir: str = "models") -> str:
    """
    Ensure the specified model is downloaded locally. 
    If not, use Hugging Face token to download it.
    Returns the full path to the model directory.
    """
    global download_stats
    token = None
    
    # Create sanitized model directory name
    sanitized_name = sanitize_model_name(model_name)
    model_dir = os.path.join(base_model_dir, sanitized_name)
    
    # Create base models directory if it doesn't exist
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Check if model is already present and complete
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "model.safetensors"
    ]
    
    # Check if model directory exists and contains all required files
    if os.path.isdir(model_dir):
        missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
        if not missing_files:
            console.print(f"[green]Model '{model_name}' is already present in '{model_dir}' folder. Skipping download.[/green]")
            return model_dir
        else:
            console.print(f"[yellow]Warning: Model directory exists but missing files: {', '.join(missing_files)}. Re-downloading...[/yellow]")
            # Optionally, backup the existing incomplete directory
            backup_dir = f"{model_dir}_incomplete_{int(time.time())}"
            shutil.move(model_dir, backup_dir)
            os.makedirs(model_dir, exist_ok=True)
    else:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    # Get Hugging Face token
    token = get_huggingface_token()
    
    # Download model and tokenizer from Hugging Face
    console.print(f"[bold yellow]Downloading model '{model_name}' from Hugging Face...[/bold yellow]")
    try:
        from huggingface_hub import snapshot_download
        
        # Define patterns for essential model files
        allow_patterns = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "model.safetensors"
        ]
        
        # Reset download stats
        download_stats = {
            "total_files": len(allow_patterns),  # Estimate of files to download
            "completed_files": 0,
            "current_file": "",
            "current_size": 0,
            "total_size": 0,
            "start_time": 0,
            "downloaded_bytes": 0,
            "download_speed": 0,
            "speed_history": []
        }
        
        # Start display thread
        display_thread = threading.Thread(target=update_download_display)
        display_thread.daemon = True
        display_thread.start()
        
        # Download model
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=allow_patterns,
            tqdm_class=None,  # Disable default tqdm progress
            max_workers=1,    # Better for tracking progress
            resume_download=True,  # Resume partial downloads
            force_download=False,  # Don't re-download files that exist
            ignore_patterns=["*.bin", "*.msgpack", "*.h5"],  # Ignore unnecessary files
            cache_dir=None,    # Use default cache
            user_agent="PipeLM/0.1.0",
            # download_callback=huggingface_download_callback  # Custom callback
        )
        
        # Update stats one final time for display
        download_stats["completed_files"] = download_stats["total_files"]
        time.sleep(0.5)  # Let the display thread show 100%

    except Exception as e:
        console.print(f"[red]Model download failed: {e}[/red]")
        sys.exit(1)
    
    # Verify model was downloaded correctly
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
    if missing_files:
        console.print(f"[red]Download incomplete. Missing required files: {', '.join(missing_files)}[/red]")
        sys.exit(1)
        
    console.print(f"[bold green]Model '{model_name}' downloaded successfully to {model_dir}.[/bold green]")
    return model_dir

def get_huggingface_token() -> str:
    """Get Hugging Face token from .env or prompt user."""
    token = None
    
    # First check environment variable
    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip():
        token = os.environ["HF_TOKEN"].strip()
        console.print("[green]Using Hugging Face token from environment variable.[/green]")
        return token
    
    if "HUGGINGFACE_HUB_TOKEN" in os.environ and os.environ["HUGGINGFACE_HUB_TOKEN"].strip():
        token = os.environ["HUGGINGFACE_HUB_TOKEN"].strip()
        console.print("[green]Using Hugging Face token from environment variable.[/green]")
        return token
    
    # Then check .env file
    env_path = ".env"
    if os.path.isfile(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_HUB_TOKEN="):
                    parts = line.split("=", 1)
                    if len(parts) == 2 and parts[1].strip():
                        token = parts[1].strip().strip("'\"")  # Remove quotes if present
                        console.print("[green]Using Hugging Face token from .env file.[/green]")
                        return token
    
    # If no token found, prompt the user
    token = getpass.getpass("Enter your Hugging Face Access Token: ").strip()
    if not token:
        console.print("[red]Error: A Hugging Face Access Token is required to download the model.[/red]")
        sys.exit(1)
    
    # Save token to .env (create or update)
    save_token_to_env(token)
    
    # Set token in environment for current session
    os.environ["HF_TOKEN"] = token
    
    return token

def save_token_to_env(token: str) -> None:
    """Save token to .env file."""
    env_path = ".env"
    new_lines = []
    token_saved = False
    
    if os.path.isfile(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN=") or line.strip().startswith("HUGGINGFACE_HUB_TOKEN="):
                    new_lines.append(f"HF_TOKEN={token}\n")
                    token_saved = True
                else:
                    new_lines.append(line)
    
    if not token_saved:
        new_lines.append(f"HF_TOKEN={token}\n")
    
    with open(env_path, "w") as f:
        f.writelines(new_lines)
    
    console.print("[green]Hugging Face token saved to .env file.[/green]")

def launch_server(model_dir: str, port: int = 8080, gpu: bool = False, gpu_layers: int = 0, quantize: str = None) -> subprocess.Popen:
    """Launch the FastAPI inference server (app.py) on the specified port. Returns the process handle."""
    console.print("[bold yellow]Starting FastAPI server...[/bold yellow]")
    
    # Set environment variables for the server
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["MODEL_DIR"] = model_dir
    
    # GPU settings
    if gpu:
        env["USE_GPU"] = "1"
        if gpu_layers > 0:
            env["GPU_LAYERS"] = str(gpu_layers)
    else:
        env["USE_GPU"] = "0"
    
    # Quantization settings
    if quantize:
        env["QUANTIZE"] = quantize
    
    # Launch the server using uvicorn in a separate process
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port), "--log-level", "warning"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        # Give server a moment to start
        time.sleep(1)
        
        # Check if process is still running
        if proc.poll() is not None:
            stderr = proc.stderr.read()
            console.print(f"[red]Server failed to start: {stderr}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        sys.exit(1)
        
    return proc

def wait_for_server(server_proc: subprocess.Popen, port: int = 8080, timeout: int = 90) -> None:
    """Wait until the server's health endpoint returns healthy or until timeout (in seconds)."""
    base_url = f"http://localhost:{port}"
    healthy = False
    
    console.print(f"[yellow]Waiting for server to be ready (timeout: {timeout}s)...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Starting model..."),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Starting model...", total=timeout)
        
        for i in range(timeout):
            progress.update(task, completed=i+1)
            
            if server_proc.poll() is not None:
                # Process ended before becoming healthy
                err_output = server_proc.stderr.read() if server_proc.stderr else "No error output"
                console.print(f"[red]Server process terminated unexpectedly. Error logs:\n{err_output}[/red]")
                sys.exit(1)
                
            try:
                r = requests.get(f"{base_url}/health", timeout=3)
                if r.status_code == 200:
                    health = r.json()
                    if health.get("status") == "healthy":
                        healthy = True
                        progress.update(task, completed=timeout)
                        break
                # If status code 503 or others, just wait and retry
            except requests.exceptions.RequestException:
                pass  # server not up yet or health not ready
                
            time.sleep(1)
    
    if not healthy:
        console.print("[red]Server did not become healthy within the timeout period.[/red]")
        server_proc.terminate()
        sys.exit(1)
        
    console.print(f"[bold green]Server is up and running on port {port}![/bold green]")
    console.print(f"[dim]API endpoints:[/dim]")
    console.print(f"[dim]  - Health check: GET http://localhost:{port}/health[/dim]")
    console.print(f"[dim]  - Model info:   GET http://localhost:{port}/[/dim]")
    console.print(f"[dim]  - Generation:   POST http://localhost:{port}/generate[/dim]")

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
            console.print("[yellow]Conversation history cleared.[/yellow]")
            continue
        elif user_input.lower() == "/info":
            try:
                info_resp = requests.get(f"{base_url}/", timeout=5)
                if info_resp.status_code == 200:
                    model_info = info_resp.json()
                    console.print("[bold blue]Model Information:[/bold blue]")
                    for key, value in model_info.items():
                        console.print(f"[blue]{key}:[/blue] {value}")
                else:
                    console.print(f"[red]Failed to get model info. Status code: {info_resp.status_code}[/red]")
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Failed to connect to server: {e}[/red]")
            continue
        elif not user_input:
            continue  # Skip empty input
            
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})
        
        # Send generation request to the local server
        payload = {"messages": messages}
        try:
            resp = requests.post(f"{base_url}/generate", json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            # Remove the last user message to allow retry
            if messages:
                messages.pop()
            continue
            
        if resp.status_code != 200:
            console.print(f"[red]Error: Server returned status code {resp.status_code}[/red]")
            if messages:
                messages.pop()  # Remove the last user message to allow retry
            continue
            
        result = resp.json()
        if "error" in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")
            # Don't add an assistant message if generation failed; allow user to retry
            if messages:
                messages.pop()
            continue
            
        full_text = result.get("generated_text", "")
        
        # Extract assistant's response from the full generated text
        assistant_response = extract_assistant_response(full_text)
        
        # Stream the assistant's response token-by-token for a more natural feel
        console.print("[bold purple]Assistant:[/bold purple]")
        
        # Check if response contains markdown formatting
        if any(marker in assistant_response for marker in ["```", "##", "_", "*", ">", "- ", "1. "]):
            # Simulate streaming with markdown content
            simulated_stream(assistant_response)
            console.print(Markdown(assistant_response))
        else:
            # Stream plain text token-by-token
            tokens = assistant_response.split()
            for i, token in enumerate(tokens):
                end_char = ' ' if i < len(tokens) - 1 else ''
                console.print(token, end=end_char)
                # Flush output and add a tiny delay to simulate streaming
                sys.stdout.flush()
                time.sleep(0.02)
            console.print()  # newline after the assistant's response
        
        # Add assistant response to history for context
        messages.append({"role": "assistant", "content": assistant_response})

def extract_assistant_response(text: str) -> str:
    """Extract the assistant's response from the generated text."""
    # Try to find the last occurrence of assistant's response
    try:
        if "\nassistant\n" in text:
            parts = text.split("\nassistant\n")
            return parts[-1].strip()
        elif "assistant:" in text.lower():
            parts = text.lower().split("assistant:")
            return parts[-1].strip()
        else:
            return text.strip()
    except Exception:
        return text.strip()

def simulated_stream(text: str) -> None:
    """Simulate streaming for markdown content."""
    for i in range(10):
        console.print(f"[dim]Generating response{'.' * (i % 4)}[/dim]", end="\r")
        time.sleep(0.1)
    console.print(" " * 30, end="\r")  # Clear the line

def check_gpu_availability() -> Tuple[bool, int]:
    """Check if GPU is available and return number of GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.device_count()
        return False, 0
    except ImportError:
        return False, 0

def main():
    parser = argparse.ArgumentParser(prog="pipelm", description="PipeLM: Chat with local Hugging Face models in the terminal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Download and chat with a specified Hugging Face model")
    run_parser.add_argument("model", type=str, help="Hugging Face model identifier (e.g. user/ModelName)")
    run_parser.add_argument("--port", "-p", type=int, default=8080, help="Port for the FastAPI server (default: 8080)")
    run_parser.add_argument("--timeout", "-t", type=int, default=90, help="Timeout in seconds for server startup (default: 90)")
    
    # GPU options
    gpu_available, gpu_count = check_gpu_availability()
    gpu_group = run_parser.add_argument_group("GPU Options")
    gpu_group.add_argument("--gpu", action="store_true", default=gpu_available, 
                          help=f"Use GPU acceleration if available (auto-detected: {'Yes' if gpu_available else 'No'})")
    gpu_group.add_argument("--no-gpu", action="store_true", help="Force CPU only mode, even if GPU is available")
    gpu_group.add_argument("--gpu-layers", type=int, default=0, 
                          help="Number of layers to offload to GPU (0=all, default: 0)")
    gpu_group.add_argument("--quantize", type=str, choices=["4bit", "8bit"], 
                          help="Quantize model weights to reduce memory usage (4bit or 8bit)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded models")
    
    # Client command for testing API
    client_parser = subparsers.add_parser("client", help="Test the API with a client request")
    client_parser.add_argument("--port", "-p", type=int, default=8080, help="Port for the FastAPI server (default: 8080)")
    client_parser.add_argument("--prompt", "-m", type=str, default="Hello, how are you?", help="Message to send to the model")
    
    args = parser.parse_args()
    
    if args.command == "run":
        model_name = args.model
        model_dir = ensure_model_available(model_name)
        
        # Handle GPU settings
        use_gpu = args.gpu and not args.no_gpu
        if use_gpu and gpu_count == 0:
            console.print("[yellow]Warning: GPU acceleration requested but no GPU detected. Falling back to CPU.[/yellow]")
            use_gpu = False
            
        if use_gpu:
            console.print(f"[green]GPU acceleration enabled. Detected {gpu_count} GPU(s).[/green]")
        else:
            console.print("[yellow]Running in CPU-only mode.[/yellow]")
        
        server_proc = launch_server(
            model_dir, 
            port=args.port, 
            gpu=use_gpu, 
            gpu_layers=args.gpu_layers,
            quantize=args.quantize
        )
        
        try:
            wait_for_server(server_proc, port=args.port, timeout=args.timeout)
            interactive_chat(base_url=f"http://localhost:{args.port}")
        except KeyboardInterrupt:
            console.print("\n[bold red]Received interrupt, shutting down...[/bold red]")
        finally:
            # Shut down the server process when done
            console.print("[yellow]Shutting down server...[/yellow]")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                console.print("[yellow]Server didn't exit gracefully, forcing shutdown...[/yellow]")
                server_proc.kill()
    
    elif args.command == "list":
        base_model_dir = "models"
        if not os.path.isdir(base_model_dir):
            console.print("[yellow]No models have been downloaded yet.[/yellow]")
            return
            
        models = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        if not models:
            console.print("[yellow]No models have been downloaded yet.[/yellow]")
            return
            
        console.print("[bold blue]Downloaded models:[/bold blue]")
        table = Table(show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Size", style="blue")
        
        for model in models:
            model_path = os.path.join(base_model_dir, model)
            required_files = ["config.json", "model.safetensors"]
            missing = [f for f in required_files if not os.path.isfile(os.path.join(model_path, f))]
            
            # Calculate model size
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    fp = os.path.join(root, file)
                    total_size += os.path.getsize(fp)
                    
            # Convert size to human-readable format
            if total_size < 1024:
                size_str = f"{total_size} B"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.1f} KB"
            elif total_size < 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
            
            status = "Incomplete" if missing else "Ready"
            table.add_row(model, status, size_str)
            
        console.print(table)
    
    elif args.command == "client":
            base_url = f"http://localhost:{args.port}"
            console.print(f"[bold blue]Testing PipeLM API at {base_url}[/bold blue]")
            
            try:
                health_resp = requests.get(f"{base_url}/health", timeout=5)
                if health_resp.status_code == 200:
                    health = health_resp.json()
                    console.print(f"[green]Server status: {health.get('status', 'unknown')}[/green]")
                    console.print(f"[green]Model: {health.get('model', 'unknown')}[/green]")
                    console.print(f"[green]Uptime: {health.get('uptime', 0):.1f} seconds[/green]")
                else:
                    console.print(f"[red]Health check failed. Status code: {health_resp.status_code}[/red]")
                    return
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Failed to connect to server: {e}[/red]")
                return
            
            # Send test message to the generate endpoint
            prompt = args.prompt
            payload = {"messages": [{"role": "user", "content": prompt}]}
            try:
                gen_resp = requests.post(f"{base_url}/generate", json=payload, timeout=60)
                if gen_resp.status_code == 200:
                    result = gen_resp.json()
                    console.print("[green]Received response from generate endpoint:[/green]")
                    console.print(result.get("generated_text", ""))
                else:
                    console.print(f"[red]Generation request failed with status code {gen_resp.status_code}[/red]")
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Generation request failed: {e}[/red]")
    
if __name__ == "__main__":
    main()