# server.py: FastAPI server for PipeLM to handle model inference
import os
import time
import subprocess
import requests
import sys
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager # Import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Request # Request might be needed if accessing app state differently
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

class Message(BaseModel):
    role: str
    content: str

class GenerationRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class HealthResponse(BaseModel):
    status: str
    model: str
    uptime: float

def format_conversation(messages: List[Message]) -> str:
    """Format the conversation history for the model."""
    formatted = ""

    # Add system message if not present
    if not messages or messages[0].role != "system":
        formatted += "system\nYou are a helpful AI assistant named PipeLM.\n\n"

    # Add all messages
    for msg in messages:
        formatted += f"{msg.role}\n{msg.content}\n\n"

    # Add the assistant tag for the model to continue
    formatted += "assistant\n"

    return formatted

# --- Lifespan function for model loading/unloading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    console.print("[cyan]Lifespan event: Startup sequence starting...[/cyan]")
    # Get model directory from environment variable
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir or not os.path.isdir(model_dir):
        console.print(f"[red]Error: Invalid model directory specified in MODEL_DIR: {model_dir}[/red]")
        raise RuntimeError(f"Invalid model directory: {model_dir}")

    console.print(f"[cyan]Loading model from {model_dir}...[/cyan]")
    start_load_time = time.time()
    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        load_time = time.time() - start_load_time
        console.print(f"[green]Model loaded successfully in {load_time:.2f} seconds![/green]")

        # Store model, tokenizer, model_dir, and start time in app state
        # app.state is the recommended place to store shared resources
        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.model_dir = model_dir
        app.state.start_time = time.time() # Record server start time *after* model loading

    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        # Propagate the error to prevent the app from starting incorrectly
        raise RuntimeError(f"Failed to load model: {e}") from e

    yield # Application runs after yield

    # --- Shutdown Logic ---
    console.print("[cyan]Lifespan event: Shutdown sequence starting...[/cyan]")
    # Clean up resources, though Pytorch/Transformers usually handle this well with device_map
    if hasattr(app.state, 'model'):
        del app.state.model
    if hasattr(app.state, 'tokenizer'):
        del app.state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("[green]Resources cleaned up. Server shutting down.[/green]")


# --- Create the FastAPI application ---
# Pass the lifespan context manager to the FastAPI constructor
app = FastAPI(title="PipeLM API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    # Access shared resources via request.app.state
    app_state = request.app.state
    if not hasattr(app_state, 'model') or not hasattr(app_state, 'tokenizer'):
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    return HealthResponse(
        status="healthy",
        model=os.path.basename(app_state.model_dir) if app_state.model_dir else "unknown",
        uptime=time.time() - app_state.start_time
    )

@app.post("/generate")
async def generate(request: Request, gen_request: GenerationRequest = Body(...)) -> Dict[str, Any]:
    # Access shared resources via request.app.state
    app_state = request.app.state
    if not hasattr(app_state, 'model') or not hasattr(app_state, 'tokenizer'):
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    model = app_state.model
    tokenizer = app_state.tokenizer

    try:
        # Format conversation history for the model
        conversation = format_conversation(gen_request.messages)

        # Tokenize the input
        inputs = tokenizer(conversation, return_tensors="pt")
        # Ensure tensors are on the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generation configuration
        generation_config = {
            "max_new_tokens": gen_request.max_tokens,
            "temperature": gen_request.temperature,
            "top_p": gen_request.top_p,
            "do_sample": gen_request.temperature > 0.0, # Sample only if temperature > 0
            "pad_token_id": tokenizer.eos_token_id
        }

        model.eval() # Set model to evaluation mode

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)

        # Decode the output, skipping special tokens (like padding or EOS)
        # Ensure decoding happens on the CPU if necessary, though decode usually handles device
        # Select the generated part, excluding the input prompt tokens
        output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

        return {"generated_text": generated_text.strip()} # Strip leading/trailing whitespace

    except Exception as e:
        console.print_exception() # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# --- Server Launch and Wait Functions (Unchanged) ---

def launch_server(model_dir: str, port: int = 8080, gpu: bool = False, gpu_layers: int = 0, quantize: str = None) -> subprocess.Popen:
    """Launch the FastAPI inference server on the specified port. Returns the process handle."""
    console.print("[bold yellow]Starting FastAPI server...[/bold yellow]")

    # Set environment variables for the server
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["MODEL_DIR"] = model_dir

    # GPU settings (These might be less relevant now device_map='auto' is used, but kept for compatibility)
    if gpu:
        env["USE_GPU"] = "1"
        if gpu_layers > 0:
            env["GPU_LAYERS"] = str(gpu_layers)
    else:
        env["USE_GPU"] = "0"

    # Quantization settings (Needs specific handling during model load if used)
    if quantize:
        env["QUANTIZE"] = quantize
        console.print("[yellow]Warning: Quantization parameter set but not explicitly handled during model load in this script.[/yellow]")

    # Get the path to the uvicorn module
    try:
        import uvicorn
        uvicorn_path = os.path.dirname(uvicorn.__file__)
        console.print(f"[dim]Using uvicorn from: {uvicorn_path}[/dim]")
    except ImportError:
        console.print("[red]Error: uvicorn package not found. Please install it with 'pip install uvicorn'.[/red]")
        sys.exit(1)

    # Find the location of the server module (assuming it's run relative to project root or installed)
    # Adjusted to be more robust - uses the name 'server' which refers to *this* file.
    server_module_name = __name__ # Gets the name of the current module ('server' if run as 'python server.py')
    if server_module_name == '__main__':
        server_module_name = 'server' # Use 'server' if run directly
    app_instance_string = f"{server_module_name}:app" # e.g., 'server:app'

    console.print(f"[dim]Server module: {server_module_name}, App instance: app[/dim]")

    # Launch the server using uvicorn in a separate process
    try:
        # Use --reload for development if needed, but remove for production
        cmd = [sys.executable, "-m", "uvicorn", app_instance_string, "--host", "0.0.0.0", "--port", str(port), "--log-level", "info"] # Changed log-level for more info
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            # Use process group for better termination control on Unix-like systems
            preexec_fn=os.setsid if os.name != "nt" else None
        )
        # Give server a moment to start (might need adjustment)
        time.sleep(3) # Increased slightly

        # Check if process started correctly
        if proc.poll() is not None:
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            console.print(f"[red]Server failed to start. Exit code: {proc.poll()}[/red]")
            console.print(f"[red]Stderr:\n{stderr}[/red]")
            console.print(f"[yellow]Stdout:\n{stdout}[/yellow]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Failed to start server process: {e}[/red]")
        sys.exit(1)

    return proc

def wait_for_server(server_proc: subprocess.Popen, port: int = 8080, timeout: int = 120) -> None: # Increased timeout
    """Wait until the server's health endpoint returns healthy or until timeout (in seconds)."""
    base_url = f"http://localhost:{port}"
    healthy = False

    console.print(f"[yellow]Waiting for server on port {port} to be ready (timeout: {timeout}s)...[/yellow]")

    progress_bar_format = "[progress.description]{task.description} [progress.percentage]{task.percentage:>3.0f}% | [progress.elapsed] elapsed | [progress.remaining] remaining"

    with Progress(
        SpinnerColumn(),
        TextColumn(progress_bar_format),
        TimeElapsedColumn(),
        # TimeRemainingColumn(), # Can be inaccurate if tasks don't progress linearly
    ) as progress:
        task = progress.add_task("[cyan]Waiting for model load...", total=timeout)

        start_wait = time.time()
        while time.time() - start_wait < timeout:
            elapsed = time.time() - start_wait
            progress.update(task, completed=elapsed)

            if server_proc.poll() is not None:
                # Process ended before becoming healthy
                stdout = server_proc.stdout.read() if server_proc.stdout else "No stdout"
                stderr = server_proc.stderr.read() if server_proc.stderr else "No stderr"
                console.print(f"\n[red]Server process terminated unexpectedly (Exit Code: {server_proc.poll()}).[/red]")
                console.print(f"[red]Stderr:\n{stderr}[/red]")
                console.print(f"[yellow]Stdout:\n{stdout}[/yellow]")
                sys.exit(1)

            try:
                # Use a slightly longer timeout for the request itself
                r = requests.get(f"{base_url}/health", timeout=5)
                if r.status_code == 200:
                    health = r.json()
                    if health.get("status") == "healthy":
                        healthy = True
                        progress.update(task, description="[green]Model ready!", completed=timeout)
                        break
                    else:
                         progress.update(task, description=f"[yellow]Health status: {health.get('status', 'unknown')}")
                elif r.status_code == 503:
                    progress.update(task, description="[yellow]Server up, model loading...")
                else:
                     progress.update(task, description=f"[yellow]Server status: {r.status_code}")

            except requests.exceptions.ConnectionError:
                progress.update(task, description="[cyan]Server not responding yet...")
                pass # Server not up yet
            except requests.exceptions.Timeout:
                 progress.update(task, description="[yellow]Health check timed out, retrying...")
                 pass # Health endpoint might be slow initially
            except requests.exceptions.RequestException as e:
                progress.update(task, description=f"[yellow]Request error: {type(e).__name__}")
                pass # Other transient errors

            time.sleep(1) # Check every second

    if not healthy:
        console.print("\n[red]Server did not become healthy within the timeout period.[/red]")
        # Try to terminate the process group on Unix-like systems
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(server_proc.pid), subprocess.signal.SIGTERM)
            else:
                server_proc.terminate()
            server_proc.wait(timeout=5) # Wait a bit for termination
        except Exception as term_err:
            console.print(f"[yellow]Could not terminate server process cleanly: {term_err}[/yellow]")
            server_proc.kill() # Force kill if termination fails
        # Read remaining output after attempting termination
        stdout = server_proc.stdout.read() if server_proc.stdout else "No stdout"
        stderr = server_proc.stderr.read() if server_proc.stderr else "No stderr"
        console.print(f"[red]Final Server Stderr:\n{stderr}[/red]")
        console.print(f"[yellow]Final Server Stdout:\n{stdout}[/yellow]")

        sys.exit(1)

    console.print(f"\n[bold green]Server is up and running on port {port}![/bold green]")
    console.print(f"[dim]API endpoints:[/dim]")
    console.print(f"[dim]  - Health check: GET http://localhost:{port}/health[/dim]")
    # Add info about root if it exists/is useful
    # console.print(f"[dim]  - Root info:    GET http://localhost:{port}/[/dim]")
    console.print(f"[dim]  - Generation:   POST http://localhost:{port}/generate[/dim]")

# # --- Main execution (Example usage) ---
# if __name__ == "__main__":
#     # This block is for direct execution/testing, e.g., python server.py
#     # It won't be run when imported by uvicorn in launch_server
#     # You would typically set MODEL_DIR via environment variable before running uvicorn
#     if "MODEL_DIR" not in os.environ:
#          print("Error: MODEL_DIR environment variable not set.")
#          print("Please set it to the path of your model directory.")
#          print("Example: export MODEL_DIR=/path/to/your/model")
#          sys.exit(1)

#     # Example: uvicorn server:app --host 0.0.0.0 --port 8080 --log-level info
#     print("To run the server, use uvicorn:")
#     print(f"MODEL_DIR={os.environ.get('MODEL_DIR')} uvicorn server:app --host 0.0.0.0 --port 8080 --log-level info")

    # Or use the launch_server/wait_for_server functions programmatically:
    # model_path = os.environ.get("MODEL_DIR")
    # server_process = None
    # try:
    #     server_process = launch_server(model_dir=model_path, port=8080)
    #     wait_for_server(server_process, port=8080)
    #     print("Server running. Press Ctrl+C to stop.")
    #     # Keep the script alive while server runs, or handle process elsewhere
    #     server_process.wait()
    # except KeyboardInterrupt:
    #     print("Stopping server...")
    # finally:
    #     if server_process and server_process.poll() is None:
    #         if os.name != "nt":
    #             os.killpg(os.getpgid(server_process.pid), subprocess.signal.SIGTERM)
    #         else:
    #             server_process.terminate()
    #         server_process.wait(timeout=10)
    #         print("Server stopped.")