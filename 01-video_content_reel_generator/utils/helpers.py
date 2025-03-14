import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel



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


import os
from datetime import datetime

def find_latest_version_directory(directory_path):
    latest_dir = None
    latest_version = None
    latest_timestamp = None

    # Iterate through all items in the directory
    for dir_name in os.listdir(directory_path):
        # Check if directory starts with 'v'
        if dir_name.startswith('v'):
            try:
                # Split the directory name into version, date, and time components
                version, date_str, time_str = dir_name.split('-')
                # Convert version to a comparable format (omit 'v')
                version_number = int(version[1:])
                # Combine date and time strings
                timestamp_str = f'{date_str} {time_str}'
                # Convert to datetime object
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d %H%M%S')

                # Update latest version if this is the first or a newer version
                if latest_version is None or version_number > latest_version:
                    latest_version = version_number
                    latest_timestamp = timestamp
                    latest_dir = dir_name
                # If same version, compare timestamps
                elif version_number == latest_version:
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_dir = dir_name
            except ValueError:
                continue  # Skip directories that don't match the format

    return latest_dir

import pandas as pd
import json

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