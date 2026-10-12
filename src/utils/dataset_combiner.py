import pandas as pd
import os
from pathlib import Path


file_path = "csv/features_dataset.csv"

def combine_datasets(input_dir, output_file):
    """
    Combines all CSV files in the specified input directory into a single CSV file.

    Parameters:
    input_dir (str): The directory containing the CSV files to combine.
    output_file (str): The path to the output CSV file.
    """
    dataframes = []

    # Iterate over all directories in the input directory
    for dir_name in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, dir_name)):
            particle_type = dir_name.split("_")[0]  
            print(f"Processing: {particle_type}")

            ds = pd.read_csv(os.path.join(input_dir, dir_name, file_path))
            ds = ds.assign(particle_type=particle_type)
            dataframes.append(ds)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"Combined dataset saved to: {output_file}")
    print(f"Shape of combined dataset: {combined_df.shape}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    input_directory = project_dir / "data/Background/Background"
    output_csv_file = project_dir / "data/Background/Engineered_Datasets/combinedbg.csv"

    combine_datasets(input_directory, output_csv_file)