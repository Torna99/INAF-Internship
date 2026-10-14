###     This script is used to visualize the moment (in terms of hits_sum) when a feture becomes non-NaN, then it
###     plots the mean value of the feature for each hits_sum value.
###     The script helps in decide how to split the original dataset in hit-sum based subsets.      

import os
import pandas as pd
import matplotlib.pyplot as plt
import math

def plot_features_vs_hits_display(df, hits_col="hits_sum", plots_per_fig=4):
    """Displays 2x2 grid figures sequentially without saving files.

    - X-axis: Mean feature value
    - Y-axis: hits_sum
    - Red dot: First hit threshold where feature is non-NaN
    - Language: English
    """

    # df_filtered = df[df[hits_col] <= max_hits].copy()
    # valid_features = [f for f in features_list if f in df_filtered.columns]

    # if not valid_features:
    #     print("No valid features found in the DataFrame.")
    #     return

    total_pages = math.ceil(df.shape[1] / plots_per_fig)

    for page in range(total_pages):
        batch = df.columns[
            page * plots_per_fig : (page + 1) * plots_per_fig
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes_flat = axes.flatten()

        for idx, feat in enumerate(batch):
            ax = axes_flat[idx]

            # Aggregate mean and count non-NaN values
            stats = (
                df.groupby(hits_col)[feat]
                .agg(mean="mean", valid_count=lambda s: s.notna().sum())
                .reset_index()
            )

            stats_valid = stats[stats["valid_count"] > 0].dropna(
                subset=["mean"]
            )

            if stats_valid.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"All values are NaN for {feat}",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                ax.set_title(feat, fontweight="bold")
                continue

            first_valid_hit = stats_valid[hits_col].iloc[0]
            first_valid_mean = stats_valid["mean"].iloc[0]

            # Mean profile line (X = feature mean, Y = hits_sum)
            ax.plot(
                stats_valid["mean"],
                stats_valid[hits_col],
                marker="o",
                color="#1f77b4",
                linewidth=2.0,
                markersize=4.5,
                label="Mean Profile",
            )

            # Highlight onset threshold
            ax.plot(
                first_valid_mean,
                first_valid_hit,
                marker="o",
                markersize=9,
                color="#d62728",
                markeredgecolor="black",
                markeredgewidth=1.2,
                label=f"Non-NaN onset (Hit = {first_valid_hit})",
            )

            ax.axhline(
                first_valid_hit,
                color="#d62728",
                linestyle=":",
                alpha=0.6,
            )

            ax.set_title(
                f"{feat} Profile vs {hits_col}", fontsize=11, fontweight="bold"
            )
            ax.set_xlabel(f"Mean Value ({feat})", fontsize=10)
            ax.set_ylabel(hits_col, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9, loc="best")

        # Remove extra empty subplots on the last figure
        for unused_idx in range(len(batch), 4):
            fig.delaxes(axes_flat[unused_idx])

        plt.suptitle(
            f"Feature Evolution vs {hits_col} (Batch {page + 1} of {total_pages})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

if __name__ == "__main__":
    # Esempio di utilizzo
    df = pd.read_csv("data/Background/Engineered_Datasets/combinedbg.csv")
    features_to_plot = [
        "E_skew",
        "E_kurt",
        "entropy",
        "gini",
        "frac_active",
        "eff_E_ge_T",
        "frac_top1",
        "frac_top3",
        "rg",
        "linearity",
        "planarity",
        "sphericity",
        "D_f",
        "f_occ",
        "L_max_hops",
        "L_max_metric",
        "N_b",
        "L_skel",
    ]
    
    plot_features_vs_hits_display(
        df=df[features_to_plot + ["hits_sum"]], 
        hits_col="hits_sum",
        plots_per_fig=4,
    )