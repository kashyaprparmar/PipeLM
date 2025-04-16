"""
downloader.py: Model download and management for PipeLM
"""
import os
import sys
import time
import shutil
import threading
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live

from pipelm.utils import sanitize_model_name, get_models_dir, format_speed, format_size, get_huggingface_token

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

def ensure_model_available(model_name: str) -> str:
    """
    Ensure the specified model is downloaded locally. 
    If not, use Hugging Face token to download it.
    Returns the full path to the model directory.
    """
    global download_stats
    token = None
    
    # Get base models directory
    base_model_dir = get_models_dir()
    
    # Create sanitized model directory name
    sanitized_name = sanitize_model_name(model_name)
    model_dir = os.path.join(base_model_dir, sanitized_name)
    
    # Check if model is already present and complete
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json"
    ]
    
    model_files_check = lambda dir_path: (
        os.path.isfile(os.path.join(dir_path, "model.safetensors")) or
        os.path.isfile(os.path.join(dir_path, "pytorch_model.bin")) or
        any(f.startswith("model-") and f.endswith(".safetensors") for f in os.listdir(dir_path)) or
        any(f.startswith("pytorch_model-") and f.endswith(".bin") for f in os.listdir(dir_path))
    )
    # Check if model directory exists and contains all required files
    if os.path.isdir(model_dir):
        missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
        if not missing_files:
            console.print(f"[green]Model '{model_name}' is already present in '{model_dir}' folder. Skipping download.[/green]")
            return model_dir
        if not model_files_check(model_dir):
            console.print(f"[red]Download incomplete. No model weights found (safetensors or bin files).[/red]")

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
            # Model files - handle both single and split files
            "model.safetensors",
            "model-*.safetensors",  # For split model files (e.g., model-00001-of-00002.safetensors)
            "model.safetensors.index.json",  # Model index file
            "pytorch_model.bin",  # For models using .bin format instead of safetensors
            "pytorch_model-*.bin",  # Split .bin model files
            "pytorch_model.bin.index.json"  # Index for .bin models
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
            # local_dir_use_symlinks=False,
            token=token,
            allow_patterns=allow_patterns,
            tqdm_class=None,  # Disable default tqdm progress
            max_workers=1,    # Better for tracking progress
            resume_download=True,  # Resume partial downloads
            force_download=False,  # Don't re-download files that exist
            ignore_patterns=["*.bin", "*.msgpack", "*.h5"],  # Ignore unnecessary files
            cache_dir=None,    # Use default cache
            user_agent="PipeLM/0.1.0",
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

def list_models() -> None:
    """List all downloaded models."""
    base_model_dir = get_models_dir()
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
        size_str = format_size(total_size)
        
        status = "Incomplete" if missing else "Ready"
        table.add_row(model, status, size_str)
        
    console.print(table)