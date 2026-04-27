import os
import pandas as pd


def convert_csv_file_to_parquet(csv_path: str, parquet_path: str) -> None:
    """Read one CSV and write Parquet with same columns. In: csv_path, parquet_path. Out: None."""
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)


def convert_folder_csv_to_parquet(input_folder: str, output_folder: str) -> None:
    """Convert all .csv in input_folder to .parquet in output_folder. In: folder paths. Out: None (prints)."""
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if not file.endswith(".csv"):
            continue
        csv_path = os.path.join(input_folder, file)
        parquet_path = os.path.join(output_folder, file.replace(".csv", ".parquet"))
        convert_csv_file_to_parquet(csv_path, parquet_path)
        print(f"Converted: {file}")
    print("FINISHED CSV TO PARQUET CONVERSION")


if __name__ == "__main__":
    input_folder = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d0_data\sic_dataset_20s_dt0p0001_csv"
    output_folder = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d0_data\sic_dataset_20s_dt0p0001_parquet"
    convert_folder_csv_to_parquet(input_folder, output_folder)
