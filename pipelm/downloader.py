# downloader.py: Model download and management for PipeLM

import os
import sys
import time
import shutil
import glob
from typing import Dict, Any, Optional, Type 
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TaskID
)
try:
    from tqdm.auto import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False

try:
    from huggingface_hub import snapshot_download, list_repo_files
    from huggingface_hub.utils import GatedRepoError
    from huggingface_hub.errors import RepositoryNotFoundError, BadRequestError, LocalEntryNotFoundError
    _hf_hub_available = True
except ImportError:
    _hf_hub_available = False

from pipelm.utils import sanitize_model_name, get_models_dir, format_size, get_huggingface_token

console = Console()

# --- Rich Progress Integration with TQDM ---
class RichTqdm(tqdm):
    """TQDM compatible class using rich.progress."""
    _current_progress: Optional[Progress] = None

    def __init__(self, *args, **kwargs):
        self.progress = RichTqdm._current_progress
        self._task_id: Optional[TaskID] = None 
        self.desc = kwargs.get("desc", "Downloading...")
        super().__init__(*args, **kwargs)

    @property
    def task_id(self) -> Optional[TaskID]:
        # Create task lazily on first access if needed
        if self._task_id is None and self.progress:
            # Try to get total from kwargs if available early, otherwise use self.total
            total_val = self.total # tqdm usually calculates this
            self._task_id = self.progress.add_task(self.desc, total=total_val)
        return self._task_id

    def display(self, msg=None, pos=None):
        # Prevent default tqdm output
        pass

    def update(self, n=1):
        # Call tqdm's update first to update internal state (like self.n)
        super().update(n)
        if self.progress and self.task_id is not None:
            current_total = self.progress.tasks[self.task_id].total
            new_total = self.total
            update_args = {"completed": self.n}
            if new_total is not None and new_total != current_total:
                update_args["total"] = new_total

            self.progress.update(self.task_id, **update_args) # type: ignore

    def close(self):
        # Ensure Rich progress reflects completion
        if self.progress and self.task_id is not None:
            is_complete = self.total is not None and self.n >= self.total
            final_description = f"[green]✔[/green] {self.desc}" if is_complete else f"[yellow]![/yellow] {self.desc}"
            self.progress.update(
                self.task_id,
                completed=self.n, # Show final count
                total=self.total,
                description=final_description,
                visible=True
                )
        # Call tqdm's close last
        super().close()


    def set_description(self, desc=None, refresh=True):
        # Update internal description
        super().set_description(desc, refresh)
        # Update Rich progress description
        self.desc = desc or "" # Store the description
        if self.progress and self._task_id is not None: # Use _task_id as task might not exist yet
             # Only update if the task has been created
             self.progress.update(self.task_id, description=self.desc)

def is_model_complete(dir_path: str) -> bool:
    """Check whether the given model directory contains required files."""
    # This internal check remains the same logic as before
    if not os.path.isdir(dir_path):
        return False

    required_files = {"config.json", "tokenizer.json"} # Core requirements
    optional_files = {"generation_config.json", "tokenizer_config.json"} # Good to have

    try:
        files_in_dir = set(os.listdir(dir_path))
    except FileNotFoundError:
        return False # Directory doesn't exist

    if not required_files.issubset(files_in_dir):
        return False

    weight_patterns = [
        "model.safetensors", "model-*.safetensors",
        "pytorch_model.bin", "pytorch_model-*.bin"
    ]
    # Use os.path.join for cross-platform compatibility in glob
    has_model_weights = any(glob.glob(os.path.join(dir_path, pattern)) for pattern in weight_patterns)

    if not has_model_weights:
         return False

    return True

def cleanup_incomplete_downloads(model_dir_base: str) -> None:
    """Remove previous incomplete download attempts for the same model."""
    incomplete_pattern = f"{model_dir_base}_incomplete_*"
    for incomplete_dir in glob.glob(incomplete_pattern):
        if os.path.isdir(incomplete_dir):
            try:
                console.print(f"[dim]Removing previous incomplete download: {incomplete_dir}[/dim]", style="yellow")
                shutil.rmtree(incomplete_dir)
            except OSError as e:
                console.print(f"[red]Error removing directory {incomplete_dir}: {e}[/red]")

def ensure_model_available(model_name: str) -> Optional[str]:
    """
    Ensure the specified model is downloaded locally using huggingface_hub.
    Shows improved progress using rich.progress.
    Returns the full path to the model directory, or None on failure.
    """
    # --- Dependency checks remain the same ---
    if not _tqdm_available:
        console.print("[red]Error: `tqdm` package not found. Please install it (`pip install tqdm`).[/red]")
        return None
    if not _hf_hub_available:
        console.print("[red]Error: `huggingface-hub` package not found. Please install it (`pip install huggingface-hub`).[/red]")
        return None

    # --- Variable setup remains the same ---
    token = get_huggingface_token()
    base_model_dir = get_models_dir()
    sanitized_name = sanitize_model_name(model_name)
    model_dir = os.path.join(base_model_dir, sanitized_name)

    # --- Model check/backup logic remains the same ---
    if is_model_complete(model_dir):
        console.print(f"[green]Model '{model_name}' is already present and appears complete in '{model_dir}'.[/green]")
        return model_dir
    if os.path.isdir(model_dir):
        console.print(f"[yellow]Model directory '{model_dir}' exists but appears incomplete or corrupted.[/yellow]")
        cleanup_incomplete_downloads(model_dir)
        backup_dir = f"{model_dir}_incomplete_{int(time.time())}"
        try:
            console.print(f"[dim]Backing up existing directory to {backup_dir}[/dim]")
            shutil.move(model_dir, backup_dir)
            os.makedirs(model_dir)
        except Exception as e:
             console.print(f"[red]Error backing up existing directory: {e}. Attempting download anyway.[/red]")
             if not os.path.exists(model_dir):
                 os.makedirs(model_dir, exist_ok=True)
    else:
        os.makedirs(base_model_dir, exist_ok=True)
        cleanup_incomplete_downloads(model_dir)
        os.makedirs(model_dir, exist_ok=True)

    progress = Progress(
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%", "•",
        DownloadColumn(), "•",
        TransferSpeedColumn(), "•",
        TimeRemainingColumn(),
        console=console,
        transient=False
    )

    console.print(f"[bold yellow]Downloading model '{model_name}' from Hugging Face...[/bold yellow]")
    download_successful = False
    try:
        RichTqdm._current_progress = progress
        with progress: # Start the Rich Progress display
             snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                # local_dir_use_symlinks=False,
                token=token,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.py", "*.md", "*.model"],
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.onnx", "*.onnx_data", "*.ckpt", "*.pt", ".gitattributes", "*.git/*"],
                force_download=True,
                max_workers=8,
                tqdm_class=RichTqdm,
                user_agent=f"PipeLM/{'0.1.0'}",
             )
             download_successful = True

    except (GatedRepoError, RepositoryNotFoundError, BadRequestError, LocalEntryNotFoundError) as e:
         # Improved error handling for repository not found or access issues
         if isinstance(e, RepositoryNotFoundError) or isinstance(e, BadRequestError):
             console.print(f"[red]Error: Repository '{model_name}' not found. Please check the model name.[/red]")
         elif isinstance(e, GatedRepoError):
             console.print(f"[red]Error: Repository '{model_name}' exists but requires authentication.[/red]")
             console.print("[yellow]Use `huggingface-cli login` to authenticate and try again.[/yellow]")
         elif isinstance(e, LocalEntryNotFoundError):
             console.print(f"[red]Error: Could not access model files for '{model_name}'.[/red]")
             console.print("[yellow]Check your internet connection and try again.[/yellow]")
         else:
             console.print(f"[red]Error accessing model repository '{model_name}': {e}[/red]")
         
         # Clean up the failed download directory
         if os.path.isdir(model_dir):
             try:
                 shutil.rmtree(model_dir)
             except OSError:
                 pass
         return None
    except Exception as e:
        import traceback
        console.print(f"[red]Model download failed unexpectedly: {e}[/red]")
        console.print("[dim]Full error traceback:[/dim]")
        traceback.print_exc()
        
        # Clean up the failed download directory
        if os.path.isdir(model_dir) and not is_model_complete(model_dir):
            try:
                shutil.rmtree(model_dir)
            except OSError:
                pass  # Ignore errors during cleanup
        return None
    finally:
        RichTqdm._current_progress = None

    if download_successful and is_model_complete(model_dir):
        console.print(f"[bold green]Model '{model_name}' downloaded successfully to {model_dir}.[/bold green]")
        return model_dir
    else:
        if download_successful:
            console.print(f"[red]Download finished, but model directory appears incomplete or corrupted.[/red]")
            console.print(f"[yellow]Location: {model_dir}[/yellow]")
        else:
            console.print(f"[red]Download failed for model '{model_name}'.[/red]")
        
        # Only clean up if directory exists and is incomplete
        if os.path.isdir(model_dir) and not is_model_complete(model_dir):
            try:
                shutil.rmtree(model_dir)
                console.print(f"[dim]Removed incomplete model directory.[/dim]")
            except OSError as e:
                console.print(f"[yellow]Could not remove incomplete model directory: {e}[/yellow]")
        
        return None


def list_models() -> None:
    """List all downloaded models, checking their status and size."""
    base_model_dir = get_models_dir()

    if not os.path.isdir(base_model_dir):
        console.print("[yellow]Model directory not found. No models downloaded yet.[/yellow]")
        return

    try:
        potential_models = [d for d in os.listdir(base_model_dir)
                            if os.path.isdir(os.path.join(base_model_dir, d))
                            and not d.startswith(sanitize_model_name(d).rsplit('_incomplete_', 1)[0] + '_incomplete_')]
    except FileNotFoundError:
         console.print("[yellow]Model directory not found. No models downloaded yet.[/yellow]")
         return
    except Exception as e:
         console.print(f"[red]Error reading model directory {base_model_dir}: {e}[/red]")
         return


    if not potential_models:
        console.print("[yellow]No models found in the models directory.[/yellow]")
        return

    console.print(f"[bold blue]Downloaded models ({base_model_dir}):[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Directory", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Size", style="blue", justify="right")

    total_size_all = 0
    for model_dir_name in sorted(potential_models):
        model_path = os.path.join(base_model_dir, model_dir_name)
        status_style = "green"
        status_icon = "✔"

        if is_model_complete(model_path):
            status = "Ready"
        else:
            status = "Incomplete"
            status_style = "yellow"
            status_icon = "⚠"

        current_size = 0
        try:
            for root, dirs, files in os.walk(model_path):
                dirs[:] = [d for d in dirs if d != '.git']
                for file in files:
                    if file == '.DS_Store': continue
                    try:
                        fp = os.path.join(root, file)
                        if os.path.exists(fp) and not os.path.islink(fp):
                             current_size += os.path.getsize(fp)
                    except OSError:
                        pass
            total_size_all += current_size
            size_str = format_size(current_size) if current_size > 0 else "-"
        except Exception:
            size_str = "[red]Error[/red]"

        table.add_row(model_dir_name, f"[{status_style}]{status_icon} {status}[/{status_style}]", size_str)

    console.print(table)
    console.print(f"Total size of listed models: [bold blue]{format_size(total_size_all)}[/bold blue]")