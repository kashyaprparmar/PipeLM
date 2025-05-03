"""
utils.py: Utility functions for PipeLM
"""
import os
import sys
import time
import getpass
import requests
from typing import Tuple, Dict, Any
from rich.console import Console
import logging
import glob
import shutil
console = Console()

def sanitize_model_name(model_name: str) -> str:
    """Convert model name to a valid directory name."""
    import re
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

def format_size(total_size: int) -> str:
    """Convert size in bytes to human-readable format."""
    if total_size < 1024:
        return f"{total_size} B"
    elif total_size < 1024 * 1024:
        return f"{total_size / 1024:.1f} KB"
    elif total_size < 1024 * 1024 * 1024:
        return f"{total_size / (1024 * 1024):.1f} MB"
    else:
        return f"{total_size / (1024 * 1024 * 1024):.2f} GB"

def get_models_dir() -> str:
    """
    Get the models directory path, creating it if it doesn't exist.
    This uses appdirs to find the proper location based on the platform.
    """
    try:
        from appdirs import user_data_dir
        base_dir = user_data_dir("pipelm", "pipelm")
    except ImportError:
        # Fallback to home directory if appdirs not available
        base_dir = os.path.join(os.path.expanduser("~"), ".pipelm")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    return models_dir

def get_huggingface_token() -> str:
    """Get Hugging Face token from environment, .env file, or prompt user."""
    token = None
    
    # First check environment variables
    for var_name in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        if var_name in os.environ and os.environ[var_name].strip():
            token = os.environ[var_name].strip()
            logging.info("[green]Using Hugging Face token from environment variable.[/green]")
            return token
    
    # Then check .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        for var_name in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
            if var_name in os.environ and os.environ[var_name].strip():
                token = os.environ[var_name].strip()
                logging.info("[green]Using Hugging Face token from .env file.[/green]")
                return token
    except ImportError:
        pass  # dotenv not installed
    
    # Check config file in user directory
    try:
        config_dir = os.path.join(os.path.dirname(get_models_dir()), "config")
        os.makedirs(config_dir, exist_ok=True)
        token_file = os.path.join(config_dir, "hf_token")
        
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                token = f.read().strip()
                if token:
                    logging.info("[green]Using Hugging Face token from config file.[/green]")
                    return token
    except Exception:
        pass  # Could not read token file
    
    # If no token found, prompt the user
    token = getpass.getpass("Enter your Hugging Face Access Token: ").strip()
    if not token:
        logging.info("[red]Error: A Hugging Face Access Token is required to download the model.[/red]")
        sys.exit(1)
    
    # Save token to config file
    try:
        config_dir = os.path.join(os.path.dirname(get_models_dir()), "config")
        os.makedirs(config_dir, exist_ok=True)
        token_file = os.path.join(config_dir, "hf_token")
        
        with open(token_file, "w") as f:
            f.write(token)
        os.chmod(token_file, 0o600)  # Secure file permissions
        logging.info("[green]Hugging Face token saved to config file.[/green]")
    except Exception as e:
        logging.info(f"[yellow]Could not save token: {e}[/yellow]")
    
    # Set token in environment for current session
    os.environ["HF_TOKEN"] = token
    
    return token

def check_gpu_availability() -> Tuple[bool, int]:
    """Check if GPU is available and return number of GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.device_count()
        return False, 0
    except ImportError:
        return False, 0

def cleanup_incomplete_downloads(model_dir_base: str) -> None:
    """
    Remove stale incomplete download directories matching
    `{model_dir_base}_incomplete_*`.
    """
    pattern = f"{model_dir_base}_incomplete_*"
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                logging.info(f"Removed incomplete download: {path}")
            except Exception as e:
                logging.warning(f"Could not remove {path}: {e}")


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
        logging.info(f"[dim]Generating response{'.' * (i % 4)}[/dim]", end="\r")
        time.sleep(0.1)
    logging.info(" " * 30, end="\r")  # Clear the line

def check_health(base_url: str):
    """
    Check server health (and implicitly GPU availability on the server).
    """
    try:
        resp = requests.get(f"{base_url}/health")
        resp.raise_for_status()
        info = resp.json()
        logging.info(f"[blue]Server status:[/blue] {info.get('status')}\n[blue]Model:[/blue] {info.get('model')}\n[blue]Uptime (s):[/blue] {info.get('uptime'):.2f}")
    except requests.RequestException as e:
        logging.info(f"[red]Health check failed:[/red] {e}")
        sys.exit(1)


def list_models() -> None:
    """
    List all downloaded models (by directory) under the models cache.
    """
    from pipelm.utils import format_size

    base = get_models_dir()
    if not os.path.isdir(base):
        print("No models directory found.")
        return

    dirs = [
        d
        for d in sorted(os.listdir(base))
        if os.path.isdir(os.path.join(base, d)) and "_backup_" not in d
    ]
    if not dirs:
        print("No models downloaded.")
        return

    for name in dirs:
        path = os.path.join(base, name)
        total = 0
        for root, _, files in os.walk(path):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
        status = "Incomplete" if "_incomplete_" in name else "Ready"
        print(f"{name:30s} {format_size(total):>10s}   [{status}]")