"""Convert simulation CSV frame dumps to Parquet."""

from __future__ import annotations

import os

import pandas as pd


def convert_csv_file_to_parquet(csv_path, parquet_path):
    """Read one CSV and write it as Parquet.

    Args:
        csv_path: Source CSV frame dump path.
        parquet_path: Destination parquet path.
    """
    pd.read_csv(csv_path).to_parquet(parquet_path, index=False)


def convert_folder_csv_to_parquet(input_folder, output_folder):
    """Convert every .csv in input_folder to .parquet in output_folder.

    Creates ``output_folder`` if needed. Each ``*.csv`` becomes a sibling-named
    ``*.parquet`` file.

    Args:
        input_folder: Directory of DEM CSV frame dumps.
        output_folder: Destination directory for parquet frames.
    """
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if not file.endswith(".csv"):
            continue
        convert_csv_file_to_parquet(
            os.path.join(input_folder, file),
            os.path.join(output_folder, file.replace(".csv", ".parquet")),
        )
        print(f"Converted: {file}")
    print("FINISHED CSV TO PARQUET CONVERSION")
