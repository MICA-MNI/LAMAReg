"""Handles loading and automatic reassembly of model files."""

import os
import shutil
from lamareg.utils.file_splitter import reassemble_file
import urllib.request

MODEL_URLS = {
    "synthseg_robust_2.0.h5": [
        "https://github.com/MICA-MNI/LAMAR-Models/raw/refs/heads/main/models/synthseg_robust_2.0.h5.000",
        "https://github.com/MICA-MNI/LAMAR-Models/raw/refs/heads/main/models/synthseg_robust_2.0.h5.001",
        "https://github.com/MICA-MNI/LAMAR-Models/raw/refs/heads/main/models/synthseg_robust_2.0.h5.002",
    ],
    "synthseg_2.0.h5": "https://github.com/MICA-MNI/LAMAR-Models/raw/refs/heads/main/models/synthseg_2.0.h5",
    "synthseg_parc_2.0.h5": "https://github.com/MICA-MNI/LAMAR-Models/raw/refs/heads/main/models/synthseg_parc_2.0.h5",
    "synthseg_qc_2.0.h5": "https://github.com/MICA-MNI/LAMAR-Models/raw/refs/heads/main/models/synthseg_qc_2.0.h5",
}


def ensure_model_file_exists(model_path):
    """Check if model file exists, and reassemble from chunks if needed.

    Args:
        model_path: Path to the model file

    Returns:
        True if the model is available (either existed or was reassembled)
    """
    if os.path.exists(model_path):
        return True

    model_name = os.path.basename(model_path)
    if model_name in MODEL_URLS:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading {model_name}...")
        try:
            if isinstance(MODEL_URLS[model_name], list):
                # If the model is split into chunks, download each chunk
                for i, url in enumerate(MODEL_URLS[model_name]):
                    chunk_path = f"{model_path}.{i:03d}"
                    urllib.request.urlretrieve(url, chunk_path)
            else:
                urllib.request.urlretrieve(MODEL_URLS[model_name], model_path)
                return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False

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
