import os
import sys
import argparse
import signal
import time
import subprocess
import getpass
from rich.console import Console
from pyfiglet import Figlet
import shutil

from huggingface_hub import snapshot_download
from pipelm.downloader import ensure_model_available, list_models, is_model_complete
from pipelm.server import launch_server, wait_for_server
from pipelm.chat import interactive_chat, send_single_message
from pipelm.utils import check_gpu_availability, check_health, get_models_dir, sanitize_model_name, get_huggingface_token
from pipelm.client import call_generate, check_health

console = Console()

# Global variable to hold the server process for signal handling
server_process_global = None

def print_banner():
    """Render and print the PipeLM banner."""
    f = Figlet(font="slant")
    banner = f.renderText("PipeLM")
    console.print(f"[cyan]{banner}[/cyan]")

def signal_handler(sig, frame):
    """Handle exit signals properly."""
    global server_process_global
    if server_process_global and server_process_global.poll() is None:
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

def login_command(token_arg:str=None)-> None:
    token = token_arg

    if token:
        os.environ['HF_TOKEN']=token
        console.print(f"[yellow]Token found in command argument.[/yellow]")
        return
    
    token = os.environ.get('HF_TOKEN')
    if token:
        console.print(f"[yellow]Token found in environment variable.[/yellow]")
        return
    
    # check if the token is in the config file
    cfg = os.path.join(os.path.dirname(get_models_dir()), 'config')
    tf = os.path.join(cfg,'hf_token')

    if os.path.exists(tf):
        with open(tf) as ef:
            for line in ef:
                # if line.startswith('HF_TOKEN='):
                token = line.strip()
                break
    if token:
        os.environ['HF_TOKEN']=token
        console.print(f"[yellow]Token found in config file {tf}[/yellow]")
        console.print("[green]Login successful.[/green]")
        return
    if not token:
        console.print("[red]No HF_TOKEN found. Please enter the HF_TOKEN[/red]")        
        token = token_arg or getpass.getpass("Enter your Hugging Face Access Token (leave blank to cancel): ").strip()
    if not token:
        console.print("[yellow]Login cancelled.[/yellow]")
        return
    try:
        subprocess.run(['huggingface-cli','login','--token',token], check=True, stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        console.print("[red]Install HF CLI: pip install 'huggingface_hub[cli]'[/red]")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Login failed (exit {e.returncode}).[/red]")
        sys.exit(1)

    # save to config and .env
    try:
        cfg = os.path.join(os.path.dirname(get_models_dir()), 'config')
        os.makedirs(cfg,exist_ok=True)
        tf = os.path.join(cfg,'hf_token')
        with open(tf,'w') as f: f.write(token)
        os.chmod(tf,0o600)
    except:
        pass
    ef = os.path.join(os.getcwd(),'.env')
    try:
        lines = []
        if os.path.exists(ef):
            with open(ef) as f: lines=[l for l in f if not l.startswith('HF_TOKEN=')]
        lines.append(f"HF_TOKEN={token}\n")
        with open(ef,'w') as f: f.writelines(lines)
    except:
        pass
    os.environ['HF_TOKEN']=token
    console.print("[yellow]Token saved to .env & config directory.[/yellow]")
    console.print("[green]Login successful.[/green]")
    return

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

    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Download command: wrapper for Hugging Face Snapshot download
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face via CLI wrapper")
    download_parser.add_argument("repo_id", help="Model name/path on Hugging Face (e.g., 'mistralai/Mistral-7B-v0.1')")
    download_parser.add_argument("filenames", nargs="*", help="Specific file names to download (e.g., config.json)", default=[])
    download_parser.add_argument("--include", nargs="*", help="Glob patterns to include (e.g., '*.safetensors')", default=[])
    download_parser.add_argument("--exclude", nargs="*", help="Glob patterns to exclude (e.g., '*.fp16.*')", default=[])
    download_parser.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model",help="Type of repository to download from")
    download_parser.add_argument("--revision", help="Specific revision (branch, tag, commit) to download from")
    download_parser.add_argument("--cache-dir", help="Cache directory for Hugging Face CLI")
    download_parser.add_argument("--local-dir", help="Target directory for downloaded files")
    download_parser.add_argument("--force-download", action="store_true", help="Force re-download of all files")
    download_parser.add_argument("--resume-download", action="store_true", help="Resume partially downloaded files")
    download_parser.add_argument("--token", help="Hugging Face token to authenticate download")
    download_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    download_parser.add_argument("--max-workers", type=int, help="Max parallel download threads")

    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded models")

    # Login command: authenticate with Hugging Face
    login_parser = subparsers.add_parser("login", help="Login to Hugging Face Hub")
    login_parser.add_argument(
        "--token", help="Hugging Face access token (prompt if omitted)"
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat with a model")
    chat_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    chat_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    chat_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    chat_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    chat_parser.add_argument("--quantize", choices=["4bit", "8bit"], help="Quantize the model")
    chat_parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    chat_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    chat_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for nucleus sampling")
    chat_parser.add_argument(
        "--model-type", choices=["text2text", "image2text"], default="text2text",
        help="Type of model to serve"
    )

    # server command
    server_parser = subparsers.add_parser("server", help="Start FastAPI server with a model")
    server_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    server_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    server_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    server_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    server_parser.add_argument("--quantize", choices=["4bit", "8bit"], help="Quantize the model")
    server_parser.add_argument("--no-stream", action="store_true", default=False, help="Enable streaming output")
    server_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    server_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    server_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for nucleus sampling")
    server_parser.add_argument(
        "--model-type", choices=["text2text", "image2text"], default="text2text",
        help="Type of model to serve"
    )

    # client command
    client_parser = subparsers.add_parser("client", help="Single message Model inference on FastAPI server")
    client_parser.add_argument("model", help="Model name/path to use (from HF or local path)")
    client_parser.add_argument("--port", type=int, default=8080, help="Port for the API server")
    client_parser.add_argument("--no-gpu", action="store_true", default=True, help="Enable GPU acceleration")
    client_parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (0=auto)")
    client_parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None, help="Quantize the model")
    client_parser.add_argument("--prompt", type=str, default="Hi", help="Input text prompt")
    client_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    client_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    client_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for nucleus sampling")
    client_parser.add_argument("--image", type=str, default="", help="Path or URL for image to be analysed")
    client_parser.add_argument("--no-stream", action="store_true", default=False, help="Enable streaming output")
    client_parser.add_argument(
        "--model-type", choices=["text2text", "image2text"], default="text2text",
        help="Type of model to serve"
    )

    args = parser.parse_args()
    # Login
    if args.command == 'login':
        login_command(args.token)
        return

    # Download with pre-check
    if args.command == 'download':
        dest = args.local_dir or os.path.join(get_models_dir(), sanitize_model_name(args.repo_id))
        # Build proxies dict

        dest = args.local_dir or os.path.join(
            get_models_dir(), sanitize_model_name(args.repo_id)
        )

        if os.path.exists(dest):
            model_dir = dest
            console.print(f"[green]Model found at {model_dir}[/green]")
            return
        token = args.token or os.environ.get("HF_TOKEN") or get_huggingface_token()

        path = ensure_model_available(
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            local_dir=dest,
            repo_type=args.repo_type,
            allow_patterns=args.include,
            ignore_patterns=args.exclude,
            force_download=args.force_download,
            resume_download=not args.resume_download,
            token=token,
            user_agent="PipeLM",
        )
        if path:
            console.print(f"[bold green]Model ready at:[/bold green] {path}")
        else:
            console.print(f"[bold red]Failed to download '{args.repo_id}'[/bold red]")
        return

        # Handle 'list' command
    if args.command == "list":
        list_models()
        return

    # Remaining commands: chat, server, client
    has_gpu, gpu_count = check_gpu_availability()
    use_gpu = has_gpu and not getattr(args, 'no_gpu', False)

    if has_gpu:
        console.print(f"[green]GPU detected: {gpu_count} GPU{'s' if gpu_count > 1 else ''}[/green]")
        if getattr(args, 'no_gpu', False):
            console.print("[yellow]GPU acceleration disabled via --no-gpu flag.[/yellow]")
        else:
            console.print("[cyan]GPU acceleration enabled.[/cyan]")
    else:
        console.print("[yellow]No GPU detected or PyTorch CUDA build not found. Running in CPU mode.[/yellow]")
        if not getattr(args, 'no_gpu', False) and args.command in ("chat", "server"):
            console.print("[yellow]Note: CPU inference can be very slow.[/yellow]")

    # Ensure model is available
    if os.path.isdir(getattr(args, 'model', '')):
        model_dir = args.model
        console.print(f"[bold]Using local model: {model_dir}[/bold]")
    else:
        console.print(f"[bold]Ensuring model '{args.model}' is available...[/bold]")
        dest = os.path.join(get_models_dir(), sanitize_model_name(args.model))
        if os.path.exists(dest):
            model_dir = dest
            console.print(f"[green]Model found at {model_dir}[/green]")
        if not dest:
            console.print(f"[red]Failed to find or download model: {args.model}[/red]")
            sys.exit(1)

    base_url = f"http://localhost:{getattr(args, 'port', 8080)}"

    if args.command == "client":
        # enable_streaming = not args.no_stream
        # try:
        #     check_health(base_url)
        #     response = send_single_message(
        #         message=args.prompt,
        #         base_url=base_url,
        #         model_type=args.model_type or ("image2text" if args.image else "text2text"),
        #         image=(args.image if args.model_type == "image2text" else ""),
        #         stream=enable_streaming,
        #         temperature=args.temperature,
        #         max_tokens=args.max_tokens,
        #         top_p=args.top_p,
        #     )
        #     console.print(f"\n[green]Final model output:[/green]\n{response}\n")
        # except Exception as e:
        #     console.print("[red]Rewrite the command correctly: pipelm client model /path/to/model --image path/to/image --prompt 'prompt' --temperature 0.7 --max-tokens 100 --top-p 0.9[/red]")
        #     console.print(f"[red]Error: {e}[/red]")
        #     sys.exit(1)
        # return
        base_url = f"http://localhost:{getattr(args, 'port', 8080)}"

        # base_url = args.port
        stream = not args.no_stream

        console.print(f"[yellow]Checking server health at {base_url}...[/yellow]")
        check_health(base_url)

        console.print(
            f"[yellow]Sending prompt: '{args.prompt}' (streaming={stream})[/yellow]\n"
        )
        messages = [{"role": "user", "content": args.prompt}]
        output = call_generate(
            base_url=base_url,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=stream,
        )
        console.print(f"\n[green]Final model output:[/green]\n{output}\n")
        return
    # Launch server for 'server' and 'chat'
    console.print(f"[cyan]Launching backend server for model '{os.path.basename(model_dir)}' on port {args.port}...[/cyan]")
    server_process_global = launch_server(
        model_dir=model_dir,
        port=args.port,
        gpu=use_gpu,
        gpu_layers=getattr(args, 'gpu_layers', 0),
        quantize=getattr(args, 'quantize', None),
        model_type=args.model_type
    )
    try:
        wait_for_server(server_process_global, args.port)
        if args.command == "server":
            print_banner()
            console.print("[yellow]Server mode: Running indefinitely. Press Ctrl+C to stop.[/yellow]")
            while server_process_global.poll() is None:
                time.sleep(1)
            console.print("[red]Server process ended unexpectedly.[/red]")
        elif args.command == "chat":
            print_banner()
            enable_streaming = not args.no_stream
            interactive_chat(base_url=base_url, streaming=enable_streaming)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Keyboard interrupt detected.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]An error occurred: {e}[/bold red]")
        signal_handler(signal.SIGTERM, None)
    finally:
        if server_process_global and server_process_global.poll() is None:
            console.print("[cyan]Ensuring server process is terminated...[/cyan]")
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(server_process_global.pid), signal.SIGTERM)
                else:
                    server_process_global.terminate()
                server_process_global.wait(timeout=5)
            except Exception as term_err:
                console.print(f"[yellow]Force killing server process: {term_err}[/yellow]")
                server_process_global.kill()
            console.print("[green]Server stopped.[/green]")

if __name__ == "__main__":
    main()