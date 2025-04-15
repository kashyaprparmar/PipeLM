"""
app.py: FastAPI server for PipeLM to handle model inference
"""
import os
import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="PipeLM API")

# Global variables to store model and tokenizer
model = None
tokenizer = None
model_dir = None

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

# Track when the server started
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, model_dir
    
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
    global model, tokenizer, model_dir, start_time
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    return {
        "status": "healthy",
        "model": os.path.basename(model_dir) if model_dir else "unknown",
        "uptime": time.time() - start_time
    }

@app.post("/generate")
async def generate(request: GenerationRequest = Body(...)) -> Dict[str, Any]:
    global model, tokenizer
    
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