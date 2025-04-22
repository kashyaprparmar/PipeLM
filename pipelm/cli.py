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
from pipelm.chat import interactive_chat, send_single_message
from pipelm.utils import check_gpu_availability, check_health

console = Console()

def signal_handler(sig, frame):
    """Handle exit signals properly."""
    # Find and terminate the server process if it exists
    global server_process_global
    if 'server_process_global' in globals() and server_process_global and server_process_global.poll() is None:
        console.print("\n[bold red]Signal received, terminating server...[/bold red]")
        try:
            if os.name != "nt":
                 os.killpg(os.getpgid(server_process_global.pid), signal.SIGTERM)
            else:
                 server_process_global.terminate()
            server_process_global.wait(timeout=5)
            console.print("[green]Server terminated.[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not terminate server cleanly: {e}. Killing...[/yellow]")
            server_process_global.kill()
    else:
        console.print("\n[bold red]Shutting down...[/bold red]")

    sys.exit(0)

# Global variable to hold the server process for signal handling
server_process_global = None

def main():
    """Main entry point for the PipeLM CLI."""
    global server_process_global

    # Set up signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="PipeLM: A lightweight API server and CLI for running LLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True) # Make command required

    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("model", help="Model name/path on Hugging Face (e.g., 'mistralai/Mistral-7B-v0.1')")

    list_parser = subparsers.add_parser("list", help="List downloaded models")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat with a model")
    chat_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    chat_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    chat_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    chat_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    chat_parser.add_argument("--quantize", choices=["4bit", "8bit"], help="Quantize the model to reduce memory usage")
    chat_parser.add_argument("--no-stream", action="store_true", help="Disable token-by-token streaming output (display full response at once)")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    chat_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to be generated")
    chat_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p value for text generation")    
    chat_parser.add_argument(
        "--model-type",
        choices=["text2text", "image2text"],
        default="text2text",
        help="Type of model to serve (text2text vs. image2text)"
    )


    # Server command
    server_parser = subparsers.add_parser("server", help="Start FastAPI server with a model")
    server_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    server_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    server_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    server_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    server_parser.add_argument("--quantize", choices=["4bit", "8bit"], help="Quantize the model to reduce memory usage")
    server_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    server_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to be generated")
    server_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p value for text generation")
    server_parser.add_argument("--no-stream", action="store_true", default=False, help="Enable token-by-token streaming output")
    server_parser.add_argument(
        "--model-type",
        choices=["text2text", "image2text"],
        default="text2text",
        help="Type of model to serve (text2text vs. image2text)"
    )
    
    # client command
    client_parser = subparsers.add_parser("client", help="Single message Model inference on FastAPI server")
    client_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    client_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    client_parser.add_argument("--no-gpu", action="store_true", default=True, help="Enable GPU acceleration")
    client_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    client_parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None, help="Quantize the model to reduce memory usage")
    client_parser.add_argument("--prompt", type=str, default="Hi", help="Input text prompt")
    client_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    client_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to be generated")
    client_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p value for text generation")
    client_parser.add_argument("--image", type=str, default="", help="Path or URL for image to be analysed")
    client_parser.add_argument("--no-stream", action="store_true", default=False, help="Enable token-by-token streaming output")

    client_parser.add_argument(
        "--model-type",
        choices=["text2text", "image2text"],
        default="text2text",
        help="Type of model to serve (text2text vs. image2text)"
    )
    
    # Handle 'help' as a subcommand to show full usage
    if len(sys.argv) == 2 and sys.argv[1] in ("help", "-h", "--help"):
        parser.print_help()
        print("\n[+] Download Command Help:\n")
        download_parser.print_help()
        print("\n[+] List Command Help:\n")
        list_parser.print_help()
        print("\n[+] Chat Command Help:\n")
        chat_parser.print_help()
        print("\n[+] Server Command Help:\n")
        server_parser.print_help()
        print("\n[+] Client Command Help:\n")
        client_parser.print_help()
        sys.exit(0)    

    args = parser.parse_args()

    # Handle download command
    if args.command == "download":
        console.print(f"[bold]Downloading model: {args.model}[/bold]")
        ensure_model_available(args.model)
        console.print("[green]Download completed successfully.[/green]")
        return # Exit after download

    # Handle list command
    if args.command == "list":
        list_models()
        return

    # server/chat logic
    # Check GPU availability
    has_gpu, gpu_count = check_gpu_availability()
    use_gpu = has_gpu and not args.no_gpu

    if has_gpu:
        console.print(f"[green]GPU detected: {gpu_count} GPU{'s' if gpu_count > 1 else ''}[/green]")
        if args.no_gpu:
            console.print("[yellow]GPU acceleration disabled via --no-gpu flag.[/yellow]")
        else:
             console.print("[cyan]GPU acceleration enabled.[/cyan]")
    else:
        console.print("[yellow]No GPU detected or PyTorch CUDA build not found. Running in CPU mode.[/yellow]")
        if not args.no_gpu and (args.command == "chat" or args.command == "server"):
            console.print("[yellow]Note: CPU inference can be very slow.[/yellow]")

    # Ensure model is available (download if necessary)
    if os.path.isdir(args.model):
        model_dir = args.model
        console.print(f"[bold]Using local model: {model_dir}[/bold]")
    else:
        console.print(f"[bold]Ensuring model '{args.model}' is available...[/bold]")
        model_dir = ensure_model_available(args.model)
        if not model_dir:
             console.print(f"[red]Failed to find or download model: {args.model}[/red]")
             sys.exit(1)

    if args.command == "client":
        enable_streaming = not args.no_stream
        try:
            # model_type = os.environ.get("MODEL_TYPE")
            # model_dir = os.environ.get("MODEL_DIR")

            base_url= f"http://localhost:{args.port}"
            check_health(base_url)

            response = send_single_message(
                message=args.prompt, 
                base_url= base_url, 
                model_type = args.model_type or ("image2text" if args.image else "text2text"),
                image= (args.image if args.model_type=="image2text" else ""),
                stream=enable_streaming, 
                temperature= args.temperature, 
                max_tokens = args.max_tokens, 
                top_p = args.top_p,
            )
            console.print(f"\n[green]Final model output:[/green] \n{response}\n")

        except Exception as e:
            console.print("[red]Rewrite the command correctly its -> pipelm client model /path/to/model --image path/to/image[/red] --prompt 'prompt' --temperature 0.7 --max-tokens 100 --top-p 0.9")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    else:
        # Launch the server (needed for both 'server' and 'chat' commands)
        console.print(f"[cyan]Launching backend server for model '{os.path.basename(model_dir)}' on port {args.port}...[/cyan]")
        server_process_global = launch_server( # Assign to global variable
            model_dir=model_dir,
            port=args.port,
            gpu=use_gpu,
            gpu_layers=args.gpu_layers,
            quantize=args.quantize,
            model_type=args.model_type
        )

        try:
            # Wait for server to be healthy
            wait_for_server(server_process_global, args.port)

            if args.command == "server":
                console.print("[yellow]Server mode: Running indefinitely. Press Ctrl+C to stop.[/yellow]")
                while server_process_global.poll() is None:
                    time.sleep(1)
                # If loop exits, server process ended unexpectedly
                console.print("[red]Server process ended unexpectedly.[/red]")

            elif args.command == "chat":
                enable_streaming = not args.no_stream
                # image_path = args.image
                interactive_chat(base_url=f"http://localhost:{args.port}", streaming=enable_streaming)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Keyboard interrupt detected.[/bold yellow]")
            # Signal handler should take over for cleanup
        except Exception as e:
            console.print(f"\n[bold red]An error occurred: {e}[/bold red]")
            # Ensure cleanup happens even on unexpected errors
            signal_handler(signal.SIGTERM, None) # Manually trigger cleanup
        finally:
            # Fallback cleanup if signal handler didn't run or failed
            if server_process_global and server_process_global.poll() is None:
                console.print("[cyan]Ensuring server process is terminated...[/cyan]")
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(server_process_global.pid), signal.SIGTERM)
                    else:
                        server_process_global.terminate()
                    server_process_global.wait(timeout=5)
                except Exception as term_err:
                    console.print(f"[yellow]Force killing server process due to error during termination: {term_err}[/yellow]")
                    server_process_global.kill()
                console.print("[green]Server stopped.[/green]")

if __name__ == "__main__":
    main()