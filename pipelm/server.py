"""
server.py: FastAPI server for PipeLM to handle model inference
"""
import os
import time
import subprocess
import requests
import sys
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Body
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

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(title="PipeLM API")
    
    # Global variables to store model and tokenizer
    model = None
    tokenizer = None
    model_dir = None
    
    # Track when the server started
    start_time = time.time()
    
    @app.on_event("startup")
    async def startup_event():
        nonlocal model, tokenizer, model_dir
        
        # Get model directory from environment variable
        model_dir = os.environ.get("MODEL_DIR")
        if not model_dir or not os.path.isdir(model_dir):
            raise RuntimeError(f"Invalid model directory: {model_dir}")
        
        print(f"Loading model from {model_dir}...")
        try:
            # Load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    @app.get("/health")
    async def health_check() -> HealthResponse:
        nonlocal model, tokenizer, model_dir, start_time
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")
        
        return {
            "status": "healthy",
            "model": os.path.basename(model_dir) if model_dir else "unknown",
            "uptime": time.time() - start_time
        }
    
    @app.post("/generate")
    async def generate(request: GenerationRequest = Body(...)) -> Dict[str, Any]:
        nonlocal model, tokenizer
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")
        
        try:
            # Format conversation history for the model
            conversation = format_conversation(request.messages)
            
            # Tokenize the input
            inputs = tokenizer(conversation, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate text
            generation_config = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": request.temperature > 0.0,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_config)
                
            # Decode and return the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {"generated_text": generated_text}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
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
    
    return app

app = create_app()

def launch_server(model_dir: str, port: int = 8080, gpu: bool = False, gpu_layers: int = 0, quantize: str = None) -> subprocess.Popen:
    """Launch the FastAPI inference server on the specified port. Returns the process handle."""
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
    
    # Get the path to the uvicorn module
    try:
        import uvicorn
        uvicorn_path = os.path.dirname(uvicorn.__file__)
        console.print(f"[dim]Using uvicorn from: {uvicorn_path}[/dim]")
    except ImportError:
        console.print("[red]Error: uvicorn package not found. Please install it with 'pip install uvicorn'.[/red]")
        sys.exit(1)
    
    # Find the location of the server module
    import pipelm.server
    server_module = os.path.abspath(pipelm.server.__file__)
    server_dir = os.path.dirname(server_module)
    console.print(f"[dim]Server module location: {server_module}[/dim]")
    
    # Launch the server using uvicorn in a separate process
    try:
        cmd = [sys.executable, "-m", "uvicorn", "pipelm.server:app", "--host", "0.0.0.0", "--port", str(port), "--log-level", "warning"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            env=env
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