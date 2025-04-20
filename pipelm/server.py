# server.py: FastAPI server for PipeLM to handle model inference
import os
import time
import subprocess
import requests
import sys
import threading
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
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
    stream: Optional[bool] = True # default to True
    image: Optional[str] = None # Optional image field

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

    formatted += "assistant\n"

    return formatted

# Lifespan function for model loading/unloading 
@asynccontextmanager
async def lifespan(app: FastAPI):
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
        # Ensure EOS token is set if not already (important for generation)
        # if tokenizer.eos_token_id is None:
        #      tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>") # Example common EOS
        #      console.print("[yellow]Warning: tokenizer.eos_token_id was not set, automatically assigning one. Check model config if issues arise.[/yellow]")
        # # Ensure PAD token is set; often same as EOS for generation
        # if tokenizer.pad_token_id is None:
        #     tokenizer.pad_token_id = tokenizer.eos_token_id
        #     console.print("[yellow]Warning: tokenizer.pad_token_id was not set, setting to eos_token_id.[/yellow]")

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        load_time = time.time() - start_load_time
        console.print(f"[green]Model loaded successfully in {load_time:.2f} seconds![/green]")

        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.model_dir = model_dir
        app.state.start_time = time.time()

    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        raise RuntimeError(f"Failed to load model: {e}") from e

    yield # Application runs after yield

    console.print("[cyan]Lifespan event: Shutdown sequence starting...[/cyan]")
    if hasattr(app.state, 'model'):
        del app.state.model
    if hasattr(app.state, 'tokenizer'):
        del app.state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("[green]Resources cleaned up. Server shutting down.[/green]")


# --- Create the FastAPI application ---
app = FastAPI(title="PipeLM API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    app_state = request.app.state
    if not hasattr(app_state, 'model') or not hasattr(app_state, 'tokenizer'):
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return HealthResponse(
        status="healthy",
        model=os.path.basename(app_state.model_dir) if app_state.model_dir else "unknown",
        uptime=time.time() - app_state.start_time
    )

# --- Modified Generate Endpoint for Streaming ---
@app.post("/generate")
async def generate(request: Request, gen_request: GenerationRequest = Body(...)):
    """
    Generates text based on the provided messages.
    Supports both streaming and non-streaming responses.
    """
    app_state = request.app.state
    if not hasattr(app_state, 'model') or not hasattr(app_state, 'tokenizer'):
        raise HTTPException(status_code=503, detail="Model is not loaded or ready.")

    model = app_state.model
    tokenizer = app_state.tokenizer

    try:
        conversation = format_conversation(gen_request.messages)
        inputs = tokenizer(conversation, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": gen_request.max_tokens,
            "temperature": gen_request.temperature,
            "top_p": gen_request.top_p,
            "do_sample": gen_request.temperature > 0.0,
            "pad_token_id": tokenizer.pad_token_id, # setting pad_token_id
            "eos_token_id": tokenizer.eos_token_id # setting eos
        }

        if gen_request.stream:
            # stream tokens
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Run generation in a separate thread to avoid blocking the FastAPI event loop
            generation_kwargs = dict(inputs, streamer=streamer, **generation_config)
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Define an async generator function to yield tokens
            async def generate_tokens() -> AsyncGenerator[str, None]:
                try:
                    for token in streamer:
                        yield token
                except Exception as e:
                    console.print(f"[red]Error during streaming generation: {e}[/red]")
                    # Optionally yield an error message or just stop
                    yield f" Error: Generation failed during streaming. {str(e)}"
                finally:
                    # Ensure thread finishes, though it should finish when streamer is exhausted
                    if thread.is_alive():
                        thread.join(timeout=1.0) # Give it a moment

            # Return the streaming response
            # Use text/plain for simple token streaming
            return StreamingResponse(generate_tokens(), media_type="text/plain")

        else:
            # --- Non-Streaming Logic ---
            model.eval()
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_config)

            output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            return {"generated_text": generated_text.strip()}

    except Exception as e:
        console.print_exception()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Server Launch 
def launch_server(model_dir: str, port: int = 8080, gpu: bool = False, gpu_layers: int = 0, quantize: str = None) -> subprocess.Popen:
    console.print("[bold yellow]Starting FastAPI server...[/bold yellow]")
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["MODEL_DIR"] = model_dir
    if gpu:
        env["USE_GPU"] = "1"
        if gpu_layers > 0:
            env["GPU_LAYERS"] = str(gpu_layers)
    else:
        env["USE_GPU"] = "0"
    if quantize:
        env["QUANTIZE"] = quantize
        console.print("[yellow]Warning: Quantization parameter set but not explicitly handled during model load in this script.[/yellow]")
    try:
        import uvicorn
        uvicorn_path = os.path.dirname(uvicorn.__file__)
        console.print(f"[dim]Using uvicorn from: {uvicorn_path}[/dim]")
    except ImportError:
        console.print("[red]Error: uvicorn package not found. Please install it with 'pip install uvicorn'.[/red]")
        sys.exit(1)

    server_module_name = __name__
    if server_module_name == '__main__':
        server_module_name = 'server'
    app_instance_string = f"{server_module_name}:app"
    console.print(f"[dim]Server module: {server_module_name}, App instance: app[/dim]")

    try:
        cmd = [sys.executable, "-m", "uvicorn", app_instance_string, "--host", "0.0.0.0", "--port", str(port), "--log-level", "info"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            preexec_fn=os.setsid if os.name != "nt" else None
        )
        time.sleep(5) # sleep allow server init

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


def wait_for_server(server_proc: subprocess.Popen, port: int = 8080, timeout: int = 180) -> None:
    base_url = f"http://localhost:{port}"
    healthy = False
    console.print(f"[yellow]Waiting for server on port {port} to be ready (timeout: {timeout}s)...[/yellow]")
    progress_bar_format = "[progress.description]{task.description} [progress.percentage]{task.percentage:>3.0f}% | [progress.elapsed] elapsed"

    with Progress(
        SpinnerColumn(),
        TextColumn(progress_bar_format),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Waiting for model load...", total=timeout)
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            elapsed = time.time() - start_wait
            progress.update(task, completed=elapsed)

            if server_proc.poll() is not None:
                stdout = server_proc.stdout.read() if server_proc.stdout else "No stdout"
                stderr = server_proc.stderr.read() if server_proc.stderr else "No stderr"
                console.print(f"\n[red]Server process terminated unexpectedly (Exit Code: {server_proc.poll()}).[/red]")
                console.print(f"[red]Stderr:\n{stderr}[/red]")
                console.print(f"[yellow]Stdout:\n{stdout}[/yellow]")
                sys.exit(1)

            try:
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
            except requests.exceptions.Timeout:
                progress.update(task, description="[yellow]Health check timed out, retrying...")
            except requests.exceptions.RequestException as e:
                 progress.update(task, description=f"[yellow]Request error: {type(e).__name__}")

            time.sleep(1)

    if not healthy:
        console.print("\n[red]Server did not become healthy within the timeout period.[/red]")
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(server_proc.pid), subprocess.signal.SIGTERM)
            else:
                server_proc.terminate()
            server_proc.wait(timeout=5)
        except Exception as term_err:
            console.print(f"[yellow]Could not terminate server process cleanly: {term_err}[/yellow]")
            server_proc.kill()
        stdout = server_proc.stdout.read() if server_proc.stdout else "No stdout"
        stderr = server_proc.stderr.read() if server_proc.stderr else "No stderr"
        console.print(f"[red]Final Server Stderr:\n{stderr}[/red]")
        console.print(f"[yellow]Final Server Stdout:\n{stdout}[/yellow]")
        sys.exit(1)

    console.print(f"\n[bold green]Server is up and running on port {port}![/bold green]")
    console.print(f"[dim]API endpoints:[/dim]")
    console.print(f"[dim] - Health check: GET http://localhost:{port}/health[/dim]")
    console.print(f"[dim] - Generation: POST http://localhost:{port}/generate[/dim]")