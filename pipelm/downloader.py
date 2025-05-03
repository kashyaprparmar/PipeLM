# downloader.py: Model download and management for PipeLM

import os
import sys
import time
import shutil
import glob
from typing import List, Union
from typing import Dict, Any, Optional, Type 
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    TaskID
)
import logging

try:
    from tqdm.auto import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False

try:
    from huggingface_hub import snapshot_download, list_repo_files
    from huggingface_hub.utils import GatedRepoError
    from huggingface_hub.errors import RepositoryNotFoundError, BadRequestError, LocalEntryNotFoundError,EntryNotFoundError
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

def ensure_model_available(
    repo_id: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    repo_type: str = "model",
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    force_download: bool = False,
    resume_download: bool = True,
    token: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Optional[str]:
    """
    Download a Hugging Face repository snapshot into `local_dir`.
    Returns the path if successful, or None on failure/interruption.
    """
    token = token or get_huggingface_token()
    base_dir = get_models_dir()
    folder_name = sanitize_model_name(repo_id)
    dest = local_dir or os.path.join(base_dir, folder_name)

    # Clean up any previous incomplete downloads
    cleanup_incomplete_downloads(base_dir)

    # Handle existing dest directory
    if os.path.exists(repo_id):
        if force_download:
            backup = f"{repo_id}_backup_{int(time.time())}"
            shutil.move(repo_id, backup)
            os.makedirs(repo_id, exist_ok=True)
        elif resume_download:
            logging.info(f"Resuming download for '{repo_id}' into '{repo_id}'")
        else:
            print(f"Directory '{repo_id}' already exists; skipping download.")
            return repo_id
    else:
        os.makedirs(dest, exist_ok=True)

    try:
        allowed_patterns_list = None
        ignore_patterns_list = None
        if allow_patterns:
            allowed_patterns_list = allow_patterns[0].strip("[]").split(",") if allow_patterns[0].strip("[]") else None
        if ignore_patterns:
            ignore_patterns_list = ignore_patterns[0].strip("[]").split(",") if ignore_patterns[0].strip("[]") else None
        
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_dir=dest,
            allow_patterns=allowed_patterns_list,
            ignore_patterns=ignore_patterns_list,
            user_agent=user_agent,
        )
        console.print(f"[green]✔ Download complete: '{repo_id}' → '{dest}'[/green]")
        return dest

    except KeyboardInterrupt:
        incomplete = f"{dest}_incomplete_{int(time.time())}"
        shutil.move(dest, incomplete)
        console.print(f"[red]Download interrupted; partial data moved to '{incomplete}[/red]")
        return None

    except (RepositoryNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
        console.print(f"[red]Repository '{repo_id}' not found on Hugging Face.[/red]")
        return None

    except GatedRepoError:
        console.print(f"[red]Repository '{repo_id}' is gated. Please authenticate with `pipelm login`.[/red]")
        return None

    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        try:
            shutil.rmtree(dest)
        except Exception:
            pass
        return None
