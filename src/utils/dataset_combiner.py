import pandas as pd
import os
from pathlib import Path


file_path = "csv/features_dataset.csv"
dir_to_ignore=["PrimaryAlphas_HelMod","PrimaryProtons_HelMod","RESULTS"]

def combine_datasets(input_dir, output_file, sample_size=None):
    """
    Combines all CSV files in the specified input directory into a single CSV file.

    Parameters:
    input_dir (str): The directory containing the CSV files to combine.
    output_file (str): The path to the output CSV file.
    """
    dataframes = []

    # Iterate over all directories in the input directory
    i = 0
    for dir_name in os.listdir(input_dir):
        if dir_name in dir_to_ignore:
            continue
        if os.path.isdir(os.path.join(input_dir, dir_name)):
            particle_type = dir_name.split("_")[0]  
            print(f"[{i}] => Processing: {particle_type} in directory: {dir_name}")

            complete_path = os.path.join(input_dir, dir_name, file_path)
            if not os.path.exists(complete_path):
                print(f"File Skipped! {complete_path}")
                continue
            ds = pd.read_csv(complete_path)
            ds = ds.assign(particle_type=particle_type)
            if sample_size is not None:
                ds = ds.sample(n=sample_size, random_state=42)
            dataframes.append(ds)
            i += 1

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print("=" * 50)
    print(f"Combined dataset saved to: {output_file}")
    print(f"Shape of combined dataset: {combined_df.shape}")
    print(f"Quantity of each particle type in the combined dataset:\n{combined_df['particle_type'].value_counts()}")




if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # input_directory = project_dir / "data/Background/Background"
    # output_csv_file = project_dir / "data/Background/Engineered_Datasets/combinedbg.csv"

    # input_directory = project_dir / "data/Background/Background_full"
    # output_csv_file = project_dir / "data/Background/Engineered_Datasets/combinedbg_full.csv"
    # combine_datasets(input_directory, output_csv_file)

    n_samples_per_tgf = 1000
    input_directory = project_dir / "data/TGF/TGF_different_directions/HELPix_TGF_LUT_H410"
    output_csv_file = project_dir / "data/Engineered_Datasets/combinedtgfH410_sampled.csv"
    combine_datasets(input_directory, output_csv_file, sample_size=n_samples_per_tgf)

    # input_directory = project_dir / "data/Background/Background_primary_fixed"
    # output_csv_file = project_dir / "data/Engineered_Datasets/bg_combined.csv"
    # combine_datasets(input_directory, output_csv_file)

    # input_directory = project_dir / "data/Background/Background_H410_LowLim10keV"
    # output_csv_file = project_dir / "data/Engineered_Datasets/bg_combined.csv"
    # combine_datasets(input_directory, output_csv_file)