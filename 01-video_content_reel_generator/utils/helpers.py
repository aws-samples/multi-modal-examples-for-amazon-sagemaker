import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Union



os.environ["HF_HOME"] = "/home/sagemaker-user/huggingface"

console = Console()


def pretty_llm_print(prompt, title=None):
    """ Generates a pretty text formatted mardown output to display LLM responses """
    # Use an f-string to handle the title more concisely
    header = f"# Model: {title}\n" if title else ""

    # Using a list to build body parts for faster concatenation
    body_parts = [header]
    for row in prompt:
        role = row['role'].capitalize()
        content = row['content']

        # Simplify handling of different content types
        body_parts.append(f"\n**{role}**:\n")
        if isinstance(content, str):
            body_parts.append(content)
        elif isinstance(content, list):
            for sub_row in content:
                sub_type = sub_row.get('type')
                if sub_type == 'image':
                    body_parts.append("/image")
                elif sub_type == 'video':
                    body_parts.append("/video")
                elif sub_type == 'text':
                    body_parts.append(sub_row.get('text', ""))

    # Join list into a single string and add double newlines for Markdown line breaks
    body = '\n\n'.join(body_parts)
    # Convert body to Markdown and wrap in a panel for stylized output
    title_markdown = Markdown(body)
    distinct_panel = Panel(title_markdown, border_style="#00FF00")
    console.print(distinct_panel)



def find_latest_version_directory(directory_path: Union[str, Path]) -> str:
    """
    Finds the latest version directory using pathlib for cross-platform safety.
    
    Version directory format: vX-YYYYMMDD-HHMMSS
    Where:
    - X = version number (integer)
    - YYYYMMDD = date of creation
    - HHMMSS = time of creation
    """
    path = Path(directory_path) if isinstance(directory_path, str) else directory_path
    latest_dir = None
    latest_version = -1
    latest_timestamp = None

    for dir_entry in path.iterdir():
        if dir_entry.is_dir() and dir_entry.name.startswith('v'):
            try:
                # Split directory name into components
                version_part, date_str, time_str = dir_entry.name.split('-', 2)
                
                # Extract version number
                version_number = int(version_part[1:])  # Remove 'v' prefix
                
                # Parse datetime
                timestamp = datetime.strptime(
                    f"{date_str} {time_str}", 
                    "%Y%m%d %H%M%S"
                )

                # Update latest version
                if (version_number > latest_version or 
                    (version_number == latest_version and 
                     timestamp > latest_timestamp)):
                    latest_version = version_number
                    latest_timestamp = timestamp
                    latest_dir = dir_entry.name
                    
            except (ValueError, IndexError):
                continue  # Skip invalid format

    if not latest_dir:
        raise FileNotFoundError(f"No valid version directories found in {path}")
        
    return str(latest_dir)



# Alternative version that reads the entire file if memory allows
def find_best_model_checkpoint(file_path):
    # Read the JSONL file
    df = pd.read_json(file_path, lines=True)
    
    # Find the last non-null best_model_checkpoint
    if 'best_model_checkpoint' in df:
        valid_checkpoints = df[df['best_model_checkpoint'].notna()]
        if not valid_checkpoints.empty:
            return valid_checkpoints.iloc[-1]['best_model_checkpoint']
    
    return None