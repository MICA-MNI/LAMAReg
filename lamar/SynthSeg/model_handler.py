"""Handles loading and automatic reassembly of model files."""

import os
import shutil
from lamar.utils.file_splitter import reassemble_file


def ensure_model_file_exists(model_path):
    """Check if model file exists, and reassemble from chunks if needed.

    Args:
        model_path: Path to the model file

    Returns:
        True if the model is available (either existed or was reassembled)
    """
    if os.path.exists(model_path):
        return True

    # Model doesn't exist, look for chunks
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)

    # Check for chunks
    chunks = [
        f
        for f in os.listdir(model_dir)
        if f.startswith(f"{model_name}.") and f[-3:].isdigit()
    ]

    if not chunks:
        print(f"ERROR: Model file {model_path} not found and no chunks detected!")
        return False

    print(
        f"Model file {model_path} not found, but {len(chunks)} chunks detected. Reassembling..."
    )
    return reassemble_file(model_path)
