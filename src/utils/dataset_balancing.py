import pandas as pd
import os
from pathlib import Path

def balance_dataset(input_ds, n_samples, target_col, random_state=42):
    parts = [
        g.sample(n=min(n_samples, len(g)), random_state=random_state)
        for _, g in input_ds.groupby(target_col)
    ]
    return pd.concat(parts, ignore_index=True)


### TEST:
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    input_ds = project_dir / "data/Background/Engineered_Datasets/combinedbg_full.csv"
    ds = pd.read_csv(input_ds)
    n_samples = 1000
    balance_feature = "particle_type"
    random_state = 42

    balanced_df = balance_dataset(ds, n_samples, balance_feature, random_state)
    print(f"Balanced dataset shape: {balanced_df.shape}")
    print(f"Quantity of each particle type in the balanced dataset:\n{balanced_df[balance_feature].value_counts()}")