import os
import glob
import pandas as pd


def particle_radius_counts_per_file(folder_path: str, size_column: str = "r") -> pd.DataFrame:
    """Stack value_counts of size_column per parquet file. In: folder_path, column name. Out: DataFrame (files x radii)."""
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files found.")

    all_counts = []
    for file in files:
        df = pd.read_parquet(file)
        counts = df[size_column].value_counts()
        counts.name = os.path.basename(file)
        all_counts.append(counts)

    return pd.DataFrame(all_counts).fillna(0)


def main() -> None:
    """Load hardcoded folder, print counts/mean/std for particle radius integrity check. In: None. Out: None."""
    folder_path = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d0_data\sic_dataset_20s_dt0p0001_parquet"
    size_column = "r"

    counts_df = particle_radius_counts_per_file(folder_path, size_column=size_column)

    print("\nCounts per file:\n")
    print(counts_df)

    print("\nAverage particle count per size (per file):\n")
    print(counts_df.mean())

    print("\nStandard deviation per size (should be ~0 if identical):\n")
    print(counts_df.std())


if __name__ == "__main__":
    main()
