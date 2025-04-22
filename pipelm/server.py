# server.py: FastAPI server for PipeLM to handle model inference
import os
import time
import subprocess
import requests
import sys
import threading
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# importing libraries for image processing
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

console = Console()

class MessageContentItem(BaseModel):
    type: str
    # Make text optional as 'image' type might not have it directly here
    text: Optional[str] = None
    # Allowing other fields potentially added by multimodal processors
    # class Config:
    #     extra = "allow" # Or define specific fields if known

class Message(BaseModel):
    role: str
    # Allow content to be a string OR a list of content items
    content: Union[str, List[MessageContentItem]]

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
    """Format the conversation history for the model (primarily for text models)."""
    formatted = ""

    if not messages or messages[0].role != "system":
        formatted += "system\nYou are a helpful AI assistant named PipeLM.\n\n"

    for msg in messages:
        content_str = ""
        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            # For text formatting, just extract the text parts
            texts = [item.text for item in msg.content if item.type == 'text' and item.text]
            content_str = " ".join(texts) # Simple join, might need adjustment
        formatted += f"{msg.role}\n{content_str}\n\n"

    formatted += "assistant\n"

    return formatted

# Lifespan function for model loading/unloading
@asynccontextmanager
async def lifespan(app: FastAPI):
    console.print("[cyan]Lifespan event: Startup sequence starting...[/cyan]")
    # Get model directory from environment variable
    model_dir = os.environ.get("MODEL_DIR")
    model_type = os.environ.get("MODEL_TYPE", "text2text")
    app.state.model_type = model_type

    if not model_dir or not os.path.isdir(model_dir):
        console.print(f"[red]Error: Invalid model directory specified in MODEL_DIR: {model_dir}[/red]")
        raise RuntimeError(f"Invalid model directory: {model_dir}")
    app.state.model_dir = model_dir

    try:
        start_load_time = time.time()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "image2text":
            console.print(f"[cyan]Loading image-to-text model from {model_dir}...[/cyan]")
            processor = AutoProcessor.from_pretrained(model_dir)
            vision_model = AutoModelForVision2Seq.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
                # device_map="auto", # Consider device placement if needed
                trust_remote_code=True
            ).to(DEVICE)

            load_time = time.time() - start_load_time
            console.print(f"[green]Model loaded successfully in {load_time:.2f} seconds![/green]")
            app.state.processor   = processor
            app.state.model = vision_model
            app.state.start_time = time.time()
        elif model_type == "text2text":
            console.print(f"[cyan]Loading text generation model from {model_dir}...[/cyan]")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                text_model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ValueError as e:
                print(f"[red]Error loading {model_type} model: {e} Change model type[/red]")
                sys.exit(1)
            text_model.eval()
            load_time = time.time() - start_load_time
            console.print(f"[green]Model loaded successfully in {load_time:.2f} seconds![/green]")

            app.state.tokenizer = tokenizer
            app.state.model = text_model
            app.state.start_time = time.time()
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        raise RuntimeError(f"Failed to load model: {e}") from e

    yield # Application runs after yield

    console.print("[cyan]Lifespan event: Shutdown sequence starting...[/cyan]")
    # ... (cleanup logic remains the same) ...
    if hasattr(app.state, 'model'):
        del app.state.model
    if hasattr(app.state, 'tokenizer'):
        del app.state.tokenizer
    if hasattr(app.state, 'processor'):
        del app.state.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("[green]Resources cleaned up. Server shutting down.[/green]")


# --- Create the FastAPI application ---
app = FastAPI(title="PipeLM API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    # ... (health check logic remains the same) ...
    app_state = request.app.state
    if not hasattr(app_state, 'model') or not (hasattr(app_state, 'tokenizer') or hasattr(app_state, 'processor')):
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
    Supports both streaming and non-streaming responses for text models.
    Supports image+text input for image models (non-streaming response).
    """
    app_state = request.app.state
    if not hasattr(app_state, 'model') or not (hasattr(app_state, 'tokenizer') or hasattr(app_state, 'processor')):
        raise HTTPException(status_code=503, detail="Model is not loaded or ready.")

    model = app_state.model

    # If we're serving an image-to-text model, run the vision2seq branch
    if app_state.model_type == "image2text" and gen_request.image:
        device = next(model.parameters()).device
        # load image from URL or local file
        try:
            console.print(f"[cyan]Loading image from: {gen_request.image}[/cyan]")
            if gen_request.image.startswith("http"):
                image_obj = load_image(gen_request.image)
            else:
                image_obj = Image.open(gen_request.image).convert("RGB")
            console.print("[green]Image loaded successfully.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load image: {e}[/red]")
            raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

        processor = app_state.processor
        # The processor.apply_chat_template should handle the structured messages
        # Ensure the template expects this format or adjust if needed based on the specific model/processor
        try:
            # Convert Pydantic models back to dicts for the processor if necessary
            messages_as_dicts = [msg.model_dump() for msg in gen_request.messages]
            console.print(f"[cyan]Applying chat template with messages: {messages_as_dicts}[/cyan]")
            prompt = processor.apply_chat_template(messages_as_dicts, add_generation_prompt=True)
            console.print(f"[cyan]Generated prompt for processor: {prompt}[/cyan]") # Be cautious logging prompts

            inputs = processor(text=prompt, images=image_obj, return_tensors="pt") # Pass single image if processor expects one
            inputs = {k: v.to(device) for k, v in inputs.items()}
            console.print("[cyan]Prepared inputs for the model.[/cyan]")
        except Exception as e:
             console.print(f"[red]Error during processor application: {e}[/red]")
             raise HTTPException(status_code=500, detail=f"Failed during text/image processing: {e}")

        try:
            with torch.no_grad():
                console.print("[cyan]Generating text from image...[/cyan]")
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_request.max_tokens
                )
            console.print("[green]Generation complete.[/green]")
            generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

            final_text = generated_text.strip()

            console.print(f"[green]Generated Text: {final_text}[/green]")
            return {"generated_text": final_text}
        except Exception as e:
            console.print(f"[red]Error during model generation or decoding: {e}[/red]")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    elif app_state.model_type == "text2text":
        try:
            tokenizer = app_state.tokenizer

            # Format conversation - ensure content is treated as string here
            conversation = format_conversation(gen_request.messages)
            inputs = tokenizer(conversation, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            generation_config = {
                "max_new_tokens": gen_request.max_tokens,
                "temperature": gen_request.temperature,
                "top_p": gen_request.top_p,
                "do_sample": gen_request.temperature > 0.0,
                # Ensure pad_token_id and eos_token_id are correctly set
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            # Add warning if pad token had to be defaulted
            if tokenizer.pad_token_id is None:
                 console.print("[yellow]Warning: tokenizer.pad_token_id was None, using eos_token_id.[/yellow]")


            if gen_request.stream:
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                generation_kwargs = dict(inputs, streamer=streamer, **generation_config)
                thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                async def generate_tokens() -> AsyncGenerator[str, None]:
                    # ... (streaming logic remains the same) ...
                    try:
                        for token in streamer:
                            yield token
                    except Exception as e:
                        console.print(f"[red]Error during streaming generation: {e}[/red]")
                        yield f" Error: Generation failed during streaming. {str(e)}"
                    finally:
                        if thread.is_alive():
                            thread.join(timeout=1.0)

                return StreamingResponse(generate_tokens(), media_type="text/plain")

            else:
                # --- Non-Streaming Text Logic ---
                model.eval()
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_config)

                # Ensure slicing is correct relative to input length
                input_token_len = inputs['input_ids'].shape[1]
                output_tokens = outputs[0][input_token_len:]
                generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                return {"generated_text": generated_text.strip()}

        except Exception as e:
            console.print_exception()
            raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
    else:
         # Should not happen if model_type is correctly set, but good practice
        raise HTTPException(status_code=400, detail=f"Unsupported request for model type '{app_state.model_type}'")


# Server Launch
def launch_server(model_dir: str, port: int = 8080, gpu: bool = False, gpu_layers: int = 0, quantize: str = None, model_type: str = "text2text") -> subprocess.Popen:
    # ... (launch_server logic remains the same) ...
    console.print("[bold yellow]Starting FastAPI server...[/bold yellow]")
    env = os.environ.copy()
    env.update({
        "PORT": str(port),
        "MODEL_DIR": model_dir,
        "MODEL_TYPE": model_type, # Pass model_type
        "GPU": str(gpu),
        "GPU_LAYERS": str(gpu_layers),
        "QUANTIZE": quantize or ""
    })

    # ... rest of launch_server ...
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
        server_module_name = 'server' # Adjust if your file structure differs
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
        time.sleep(5) # Allow server time to initialize

        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else "No stdout"
            stderr = proc.stderr.read() if proc.stderr else "No stderr"
            console.print(f"[red]Server failed to start. Exit code: {proc.poll()}[/red]")
            console.print(f"[red]Stderr:\n{stderr}[/red]")
            console.print(f"[yellow]Stdout:\n{stdout}[/yellow]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Failed to start server process: {e}[/red]")
        sys.exit(1)

    return proc


def wait_for_server(server_proc: subprocess.Popen, port: int = 8080, timeout: int = 180) -> None:
    # ... (wait_for_server logic remains the same) ...
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
        # ... (cleanup on timeout remains the same) ...
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