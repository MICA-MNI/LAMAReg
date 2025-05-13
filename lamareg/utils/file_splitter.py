"""Utility for splitting large files into chunks and reassembling them."""

import os
import shutil


def split_file(input_path, chunk_size_mb=95, output_dir=None):
    """Split a large file into chunks under the specified size.

    Args:
        input_path: Path to the file to split
        chunk_size_mb: Maximum size of each chunk in MB
        output_dir: Directory to store chunks (default: same as input file)

    Returns:
        List of chunk paths
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    base_name = os.path.basename(input_path)
    chunk_paths = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "rb") as f_in:
        chunk_num = 0
        while True:
            chunk_data = f_in.read(chunk_size)
            if not chunk_data:
                break

            chunk_path = os.path.join(output_dir, f"{base_name}.{chunk_num:03d}")
            chunk_paths.append(chunk_path)

            with open(chunk_path, "wb") as f_out:
                f_out.write(chunk_data)

            print(f"Created chunk {chunk_num}: {chunk_path}")
            chunk_num += 1

    return chunk_paths


def reassemble_file(output_path, pattern=None, chunks_dir=None):
    """Reassemble file from chunks.

    Args:
        output_path: Path where the reassembled file will be saved
        pattern: Base pattern to look for chunks. If None, uses output_path basename.
        chunks_dir: Directory containing chunks. If None, uses output_path directory.

    Returns:
        True if reassembly was successful, False otherwise
    """
    if chunks_dir is None:
        chunks_dir = os.path.dirname(output_path)

    if pattern is None:
        pattern = os.path.basename(output_path)

    # Get all chunks sorted by number
    chunks = [f for f in os.listdir(chunks_dir) if f.startswith(pattern) and "." in f]
    chunks.sort(key=lambda x: int(x.split(".")[-1]))

    if not chunks:
        print(f"No chunks found matching pattern {pattern} in {chunks_dir}")
        return False

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Reassemble the file
    with open(output_path, "wb") as f_out:
        for chunk in chunks:
            chunk_path = os.path.join(chunks_dir, chunk)
            with open(chunk_path, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)

    print(f"Successfully reassembled {output_path} from {len(chunks)} chunks")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split or reassemble large files")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split a file into chunks")
    split_parser.add_argument("input_path", help="Path to the file to split")
    split_parser.add_argument(
        "--size", type=int, default=95, help="Maximum size of each chunk in MB"
    )
    split_parser.add_argument("--output-dir", help="Directory to store chunks")

    # Reassemble command
    reassemble_parser = subparsers.add_parser(
        "reassemble", help="Reassemble chunks into a file"
    )
    reassemble_parser.add_argument(
        "output_path", help="Path where the reassembled file will be saved"
    )
    reassemble_parser.add_argument("--pattern", help="Base pattern to look for chunks")
    reassemble_parser.add_argument("--chunks-dir", help="Directory containing chunks")

    args = parser.parse_args()

    if args.command == "split":
        split_file(args.input_path, args.size, args.output_dir)
    elif args.command == "reassemble":
        reassemble_file(args.output_path, args.pattern, args.chunks_dir)
    else:
        parser.print_help()
