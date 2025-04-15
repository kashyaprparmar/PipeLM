"""
cli.py: Command-line interface for PipeLM
"""
import os
import sys
import argparse
import signal
import time
from rich.console import Console

from pipelm.downloader import ensure_model_available, list_models
from pipelm.server import launch_server, wait_for_server
from pipelm.chat import interactive_chat
from pipelm.utils import check_gpu_availability

console = Console()

def signal_handler(sig, frame):
    """Handle exit signals properly."""
    console.print("\n[bold red]Shutting down...[/bold red]")
    sys.exit(0)

def main():
    """Main entry point for the PipeLM CLI."""
    # Set up signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="PipeLM: A lightweight API server and CLI for running LLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("model", help="Model name/path on Hugging Face (e.g., 'mistralai/Mistral-7B-v0.1')")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded models")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat with a model")
    chat_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    chat_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    chat_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    chat_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    chat_parser.add_argument("--quantize", choices=["4bit", "8bit"], help="Quantize the model to reduce memory usage")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start FastAPI server with a model")
    server_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    server_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    server_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    server_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    server_parser.add_argument("--quantize", choices=["4bit", "8bit"], help="Quantize the model to reduce memory usage")
    
    # Parse arguments
    args = parser.parse_args()
    
    # No command provided, show help
    if not args.command:
        parser.print_help()
        return
    
    # Handle download command
    if args.command == "download":
        console.print(f"[bold]Downloading model: {args.model}[/bold]")
        ensure_model_available(args.model)
        console.print("[green]Download completed successfully.[/green]")
        return
    
    # Handle list command
    if args.command == "list":
        list_models()
        return
    
    # Check for GPU availability
    has_gpu, gpu_count = check_gpu_availability()
    if has_gpu:
        console.print(f"[green]GPU detected: {gpu_count} GPU{'s' if gpu_count > 1 else ''}[/green]")
    else:
        console.print("[yellow]No GPU detected. Running in CPU mode.[/yellow]")
        if not args.no_gpu and (args.command == "chat" or args.command == "server"):
            console.print("[yellow]Note: CPU inference can be very slow.[/yellow]")
    
    # Handle server command
    if args.command == "server":
        # Check if model is local path or needs to be downloaded
        if os.path.isdir(args.model):
            model_dir = args.model
            console.print(f"[bold]Using local model: {model_dir}[/bold]")
        else:
            model_dir = ensure_model_available(args.model)
        
        # Launch the server
        use_gpu = has_gpu and not args.no_gpu
        server_proc = launch_server(
            model_dir=model_dir, 
            port=args.port, 
            gpu=use_gpu, 
            gpu_layers=args.gpu_layers, 
            quantize=args.quantize
        )
        
        try:
            # Wait for server to be healthy
            wait_for_server(server_proc, args.port)
            
            # Keep the server running until interrupted
            console.print("[yellow]Press Ctrl+C to stop the server...[/yellow]")
            while server_proc.poll() is None:
                time.sleep(1)
                
        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping server...[/bold red]")
        finally:
            if server_proc.poll() is None:
                server_proc.terminate()
                server_proc.wait(timeout=5)
        
        return
    
    # Handle chat command
    if args.command == "chat":
        # Check if model is local path or needs to be downloaded
        if os.path.isdir(args.model):
            model_dir = args.model
            console.print(f"[bold]Using local model: {model_dir}[/bold]")
        else:
            model_dir = ensure_model_available(args.model)
        
        # Launch the server
        use_gpu = has_gpu and not args.no_gpu
        server_proc = launch_server(
            model_dir=model_dir, 
            port=args.port, 
            gpu=use_gpu, 
            gpu_layers=args.gpu_layers, 
            quantize=args.quantize
        )
        
        try:
            # Wait for server to be healthy
            wait_for_server(server_proc, args.port)
            
            # Start interactive chat
            interactive_chat(base_url=f"http://localhost:{args.port}")
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping chat and server...[/bold red]")
        finally:
            if server_proc.poll() is None:
                server_proc.terminate()
                server_proc.wait(timeout=5)
        
        return

if __name__ == "__main__":
    main()