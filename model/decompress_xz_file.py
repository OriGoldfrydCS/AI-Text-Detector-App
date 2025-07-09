import os
import lzma
import shutil
import argparse

def decompress_xz_file(input_path: str, output_path: str):
    """
    Decompress an .xz-compressed file to its original form.

    Args:
        input_path (str): Path to the .xz file.
        output_path (str): Path where the decompressed file will be written.
    """
    # Open the .xz file for reading in binary mode
    with lzma.open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        # Copy the decompressed bytes to the output file
        shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed '{input_path}' -> '{output_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Decompress one or more .xz-compressed files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to .xz files to decompress"
    )
    args = parser.parse_args()

    for xz_path in args.files:
        # Skip files that don't end with .xz
        if not xz_path.lower().endswith(".xz"):
            print(f"Skipping '{xz_path}': not an .xz file")
            continue

        # Derive the output filename by stripping the .xz extension
        output_path = xz_path[:-3]
        # Create parent directories for output if needed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Perform decompression
        decompress_xz_file(xz_path, output_path)

if __name__ == "__main__":
    main()
