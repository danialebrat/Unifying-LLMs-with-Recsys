import pandas as pd
import seaborn as sns
from RecEvaluator import RecEvaluator


def load_recommendations(path_dict):
    """
    Load recommendation results from different recommender systems.

    Args:
        path_dict (dict): A dictionary where keys are recommender names and values are file paths.

    Returns:
        dict: A dictionary with recommender names as keys and their corresponding DataFrames as values.
    """
    recommendations = {}
    for name, path in path_dict.items():
        try:
            recommendations[name] = pd.read_csv(path)
            print(f"Loaded recommendations for {name} from {path}.")
        except Exception as e:
            print(f"Error loading recommendations for {name} from {path}: {e}")
    return recommendations


def evaluate_recommenders(test_df, recommendations_dict, k_values, evaluator_params):
    """
    Evaluate multiple recommender systems and collect their performance metrics.

    Args:
        test_df (pd.DataFrame): The test dataset.
        recommendations_dict (dict): Dictionary of recommender names and their recommendation DataFrames.
        k_values (list): List of K values to evaluate.
        evaluator_params (dict): Parameters required to instantiate RecEvaluator.

    Returns:
        pd.DataFrame: A consolidated DataFrame containing evaluation metrics for all recommenders.
    """
    all_results = []

    for name, rec_df in recommendations_dict.items():
        print(f"Evaluating recommender: {name}")
        # Instantiate the evaluator for the current recommender
        rec_evaluator = RecEvaluator(
            test_df=test_df,
            recommendations_df=rec_df,
            user_id_col=evaluator_params['user_id_col'],
            item_id_col=evaluator_params['item_id_col'],
            rating_col=evaluator_params['rating_col'],
            rank_col=evaluator_params['rank_col'],
            relevance_threshold=evaluator_params['relevance_threshold']
        )

        # Compute all metrics for the current recommender
        results_df = rec_evaluator.evaluate_all_metrics(k_values)
        results_df['Recommender'] = name  # Add a column to identify the recommender

        all_results.append(results_df)
        print(f"Completed evaluation for {name}.")

    # Concatenate all results into a single DataFrame
    consolidated_results = pd.concat(all_results, ignore_index=True)
    return consolidated_results


def plot_all_metrics(consolidated_df, metrics, save_path):
    """
    Plot evaluation metrics for all recommenders.

    Args:
        consolidated_df (pd.DataFrame): The consolidated evaluation results.
        metrics (list): List of metric names to plot.
        save_path (str): Path to save the plot image.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Melt the DataFrame for easier plotting with seaborn
    melted_df = consolidated_df.melt(id_vars=['Recommender', 'K'], value_vars=metrics,
                                     var_name='Metric', value_name='Value')

    # Create a FacetGrid to plot each metric separately
    g = sns.FacetGrid(melted_df, col="Metric", hue="Recommender", sharey=False, height=4, aspect=1.2)
    g.map(sns.lineplot, "K", "Value", marker="o")
    g.add_legend()
    g.set_titles("{col_name} @ K")
    g.set_axis_labels("K", "Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved evaluation plot to {save_path}.")


def main():
    data = "1m"
    Path = f"Dataset/{data}/train_test_sets"

    # Load the ratings data
    trainset_df = pd.read_csv(f'{Path}/u1.base',
                              sep='\t',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'],
                              encoding='latin-1')

    # testset_df = pd.read_csv(f'{Path}/u1.test',
    #                          sep='\t',
    #                          names=['user_id', 'movie_id', 'rating', 'timestamp'],
    #                          encoding='latin-1')

    testset_df = pd.read_csv(f'Lusifer/{data}_test_set/enriched_test_set_{data}.csv')


    # ---------------------------------------------------------------------
    # Load recommendation results for all recommenders
    recommenders_paths = {
        "HGCF": f"output/{data}/vgcf_recommendations_{data}.csv",
        "LightGCN": f"output/{data}/lightgcn_result_{data}.csv",
        "NGCF": f"output/{data}/ngcf_result_{data}.csv",
        "ALS": f"output/{data}/als_result_{data}.csv"
        # "GAT": f"output/{data}/gat_recommendations_{data}.csv"
    }
    recommendations_dict = load_recommendations(recommenders_paths)
    # ---------------------------------------------------------------------

    # Define evaluator parameters
    evaluator_params = {
        'user_id_col': "user_id",
        'item_id_col': "movie_id",
        'rating_col': "rating",  # Column in 'testset_df' for ratings
        'rank_col': "recommendation_rank",  # Column in 'recommendations' for ranking
        'relevance_threshold': 4.0  # Typically 4.0+ means "relevant" in MovieLens
    }

    # Define K values and metrics to evaluate
    k_values = [1, 5, 10, 20, 30, 40, 50]
    metrics_to_plot = ["Precision", "Recall", "F1", "NDCG", "MRR", "MAP", "ItemCoverage", "Novelty", "ItemFairness"]


# Evaluate all recommenders
    consolidated_results = evaluate_recommenders(
        test_df=testset_df,
        recommendations_dict=recommendations_dict,
        k_values=k_values,
        evaluator_params=evaluator_params
    )

    # Reorder columns for better readability
    columns_order = ['Recommender', 'K'] + metrics_to_plot
    consolidated_results = consolidated_results[columns_order]

    # 5. Print or inspect the consolidated results
    print("Consolidated Evaluation Results:\n", consolidated_results)

    # 6. Plot the metrics for all recommenders and store the plot
    plot_all_metrics(
        consolidated_df=consolidated_results,
        metrics=metrics_to_plot,
        save_path="evaluation_plot_comparison.png"
    )

    # 7. Save the consolidated numeric results to a CSV file
    consolidated_results.to_csv("evaluation_results_comparison.csv", index=False)
    print("Saved consolidated evaluation results to evaluation_results_comparison.csv.")


if __name__ == "__main__":
    main()
