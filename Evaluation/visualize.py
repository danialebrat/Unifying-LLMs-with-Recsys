# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
#
#
# def plot_all_metrics(consolidated_dfs,
#                      metrics=['Precision', 'Recall', 'NDCG', 'MAP', 'ItemCoverage', 'Novelty'],
#                      save_path='evaluation_plots.svg'):
#     """
#     Plot evaluation metrics for all recommenders across multiple datasets.
#
#     For each dataset (e.g., 'MovieLens 100k' and 'MovieLens 1m'),
#     a 2x3 grid of subplots (one per metric) is created. All grids are arranged
#     vertically in a single figure. This function assumes that each DataFrame in
#     consolidated_dfs contains at least the following columns:
#     ['Recommender', 'K', 'Precision', 'Recall', 'NDCG', 'MAP', 'ItemCoverage', 'Novelty'].
#
#     Args:
#         consolidated_dfs (dict): Dictionary where keys are dataset names (e.g.,
#                                  'MovieLens 100k', 'MovieLens 1m') and values are
#                                  pandas DataFrames containing evaluation metrics.
#         metrics (list): List of metric names to plot.
#         save_path (str): Path to save the SVG plot.
#
#     Example:
#         consolidated_dfs = {
#             'MovieLens 100k': df_100k,
#             'MovieLens 1m': df_1m
#         }
#         plot_all_metrics(consolidated_dfs, save_path='evaluation_plots.svg')
#     """
#
#     # Set a high-quality theme for scientific publications
#     sns.set_theme(style="whitegrid", context="talk")
#
#     # Ensure consistent colors for each recommender across datasets.
#     # Compute the union of all recommender names.
#     all_recommenders = set()
#     for df in consolidated_dfs.values():
#         all_recommenders.update(df['Recommender'].unique())
#     all_recommenders = sorted(list(all_recommenders))
#     # Create a palette mapping each recommender to a color.
#     palette = dict(zip(all_recommenders, sns.color_palette("Set2", n_colors=len(all_recommenders))))
#
#     n_datasets = len(consolidated_dfs)
#     n_metrics = len(metrics)
#
#     # For each dataset we create a 2x3 grid (6 subplots). The overall grid will have:
#     #    total rows = 2 * n_datasets, and 3 columns.
#     fig, axes = plt.subplots(nrows=2 * n_datasets, ncols=3, figsize=(18, 10 * n_datasets))
#     axes = np.array(axes).reshape(2 * n_datasets, 3)
#
#     # Variable to store legend handles and labels from the first subplot
#     legend_handles, legend_labels = None, None
#
#     # Loop through each dataset and plot all metrics.
#     for d_idx, (dataset_name, df) in enumerate(consolidated_dfs.items()):
#         # For each dataset, determine the row offset.
#         # (E.g., for the first dataset, rows 0-1; for the second, rows 2-3.)
#         row_offset = d_idx * 2
#         for i, metric in enumerate(metrics):
#             ax = axes[row_offset + (i // 3), i % 3]
#             sns.lineplot(
#                 data=df,
#                 x='K', y=metric,
#                 hue='Recommender',
#                 palette=palette,
#                 marker='o',
#                 ax=ax
#             )
#             ax.set_title(f"{metric} ({dataset_name})", fontsize=16, pad=12)
#             ax.set_xlabel("K", fontsize=14)
#             ax.set_ylabel("Score", fontsize=14)
#             ax.tick_params(axis='both', which='major', labelsize=12)
#
#             # Capture legend handles/labels from the very first subplot only.
#             if d_idx == 0 and i == 0:
#                 legend_handles, legend_labels = ax.get_legend_handles_labels()
#             # Remove individual legends to avoid clutter.
#             ax.get_legend().remove()
#
#     # Create one global legend at the bottom of the figure.
#     fig.legend(legend_handles, legend_labels,
#                loc='lower center', ncol=len(legend_labels),
#                frameon=False, fontsize=12)
#
#     plt.tight_layout(rect=[0, 0.05, 1, 1])
#     plt.savefig(save_path, format='svg', dpi=300)
#     plt.close(fig)
#     print(f"Saved evaluation plots to {save_path} (SVG format).")
#
#
# if __name__ == "__main__":
#
#     movielens_100k = pd.read_csv("evaluation_results_comparison_100k.csv")
#     movielens_1m = pd.read_csv("evaluation_results_comparison_1m.csv")
#
#     consolidated_dfs = {
#         'MovieLens 100k': movielens_100k,
#         'MovieLens 1m': movielens_1m
#     }
#     plot_all_metrics(consolidated_dfs, save_path='evaluation_plots_all.svg')
#
#
#


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def plot_all_metrics_overlay(consolidated_dfs,
                             metrics=['Precision', 'Recall', 'NDCG', 'MAP', 'ItemCoverage', 'Novelty'],
                             save_path='evaluation_plots_overlay.svg'):
    """
    Plot evaluation metrics for all recommenders with overlaid results from multiple datasets.

    Instead of plotting separate grids per dataset, this function overlays the results from
    each dataset in a single plot per metric. Recommenders are differentiated by color and datasets
    are differentiated by marker and line style.

    Args:
        consolidated_dfs (dict): Dictionary where keys are dataset names (e.g., 'MovieLens 100k',
                                 'MovieLens 1m') and values are pandas DataFrames containing
                                 evaluation metrics. Each DataFrame must include the columns:
                                 ['Recommender', 'K', 'Precision', 'Recall', 'NDCG', 'MAP',
                                  'ItemCoverage', 'Novelty'].
        metrics (list): List of metric names to plot.
        save_path (str): Path to save the SVG plot.

    Example:
        consolidated_dfs = {
            'MovieLens 100k': df_100k,
            'MovieLens 1m': df_1m
        }
        plot_all_metrics_overlay(consolidated_dfs, save_path='evaluation_plots_overlay.svg')
    """
    # Set a high-quality theme for scientific publications.
    sns.set_theme(style="whitegrid", context="talk")

    # Get the union of all recommenders across datasets for a consistent color palette.
    all_recommenders = set()
    for df in consolidated_dfs.values():
        all_recommenders.update(df['Recommender'].unique())
    all_recommenders = sorted(list(all_recommenders))
    palette = dict(zip(all_recommenders, sns.color_palette("Set2", n_colors=len(all_recommenders))))

    # Combine all datasets into one DataFrame, adding a 'Dataset' column.
    df_combined = pd.concat([df.assign(Dataset=ds) for ds, df in consolidated_dfs.items()],
                            ignore_index=True)

    # Define marker and dash styles to differentiate datasets.
    datasets = list(consolidated_dfs.keys())
    # For two datasets we choose specific styles:
    if len(datasets) == 2:
        markers = {datasets[0]: "o", datasets[1]: "s"}
        # (1, 0) represents a solid line; (4, 2) will be rendered as dashed.
        dashes = {datasets[0]: (1, 0), datasets[1]: (4, 2)}
    else:
        # For >2 datasets, assign from lists (extend as needed)
        marker_list = ['o', 's', 'D', '^', 'v', '<', '>']
        dash_list = [(1, 0), (4, 2), (2, 2), (5, 2), (3, 1)]
        markers = {ds: marker_list[i % len(marker_list)] for i, ds in enumerate(datasets)}
        dashes = {ds: dash_list[i % len(dash_list)] for i, ds in enumerate(datasets)}

    # Determine grid dimensions. For instance, for 6 metrics we'll use a 2x3 grid.
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    # Ensure axes is a flat array (even if there's only one subplot)
    if n_metrics > 1:
        axes = np.array(axes).flatten()
    else:
        axes = [axes]

    # Plot each metric, overlaying the results from each dataset.
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(data=df_combined,
                     x='K',
                     y=metric,
                     hue='Recommender',
                     style='Dataset',
                     markers=markers,
                     dashes=dashes,
                     palette=palette,
                     ax=ax,
                     legend=False)
        ax.set_title(metric, fontsize=16, pad=12)
        ax.set_xlabel("K", fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Remove any extra axes.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # --- Create global legends ---
    # Legend for recommenders (colors)
    recommender_handles = [Line2D([], [], color=palette[r],
                                  marker='o', linestyle='-', markersize=8,
                                  label=r)
                           for r in all_recommenders]

    # For datasets, we convert the dash tuples to simple string representations.
    # Here, (1, 0) is rendered as a solid line, and any other value will be shown as dashed.
    dataset_handles = []
    for ds in datasets:
        ls = '-' if dashes[ds] == (1, 0) else '--'
        dataset_handles.append(Line2D([], [], color='black',
                                      marker=markers[ds],
                                      linestyle=ls,
                                      markersize=8,
                                      label=ds))

    # Add the legends manually. To display two separate legends we add the first and then
    # add the second as an artist.
    legend1 = fig.legend(handles=recommender_handles,
                         title="Recommender",
                         loc='upper center',
                         ncol=len(recommender_handles),
                         bbox_to_anchor=(0.5, 0.97),
                         frameon=False,
                         fontsize=12,
                         title_fontsize=12)

    legend2 = fig.legend(handles=dataset_handles,
                         title="Dataset",
                         loc='upper center',
                         ncol=len(dataset_handles),
                         bbox_to_anchor=(0.5, 0.91),
                         frameon=False,
                         fontsize=12,
                         title_fontsize=12)
    fig.add_artist(legend1)  # ensure the first legend stays

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(save_path, format='svg', dpi=300)
    plt.close(fig)
    print(f"Saved evaluation plots to {save_path} (SVG format).")


if __name__ == "__main__":
    # Load your CSV files.
    movielens_100k = pd.read_csv("evaluation_results_comparison_100k.csv")
    movielens_1m = pd.read_csv("evaluation_results_comparison_1m.csv")

    consolidated_dfs = {
        'MovieLens 100k': movielens_100k,
        'MovieLens 1m': movielens_1m
    }
    plot_all_metrics_overlay(consolidated_dfs, save_path='evaluation_plots_overlay.svg')

