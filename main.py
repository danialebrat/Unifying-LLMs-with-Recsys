import pandas as pd

# from VGCF import VGCF
# from Vectorized_Clustered_KNN import ClusteredKNN
# from UserBasedRecommender import UserBasedRecommender
# from VNMF import VNMF
from embedding_layer import EmbeddingLayer
from RecEvaluator import RecEvaluator
# from baselines.NGCF import NGCF

if __name__ == "__main__":
    Path = "Dataset/100k/train_test_sets"

    # Load the ratings data
    trainset_df = pd.read_csv(f'{Path}/u1.base',
                            sep='\t',
                            names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            encoding='latin-1')


    testset_df = pd.read_csv(f'{Path}/u1.test',
                            sep='\t',
                            names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            encoding='latin-1')


    users_df = pd.read_csv("Dataset/100k/users_with_summary_df.csv")

    content_df = pd.read_pickle("Dataset/100k/movies_enriched_dataset.pkl")

    embedd = EmbeddingLayer()

    users_df = embedd.convert_to_embeddings(df=users_df, col="summary")
    content_df = embedd.convert_to_embeddings(df=content_df, col="movie_info")

    # ------------------------------------------------------------
    # clustered_KNN


    # recommender = ClusteredKNN(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    # recommender.set_column_names(user_id_column="user_id",
    #                             content_id_column="movie_id",
    #                             user_attribute_column="summary_vector",
    #                             content_attribute_column="movie_info_vector")
    #
    # knn_recommendations = recommender.get_recommendations()
    # print("done")

    # ------------------------------------------------------------
    # user_based profile

    # recommender = UserBasedRecommender(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    # recommender.set_column_names(user_id_column="user_id",
    #                             content_id_column="movie_id",
    #                             user_attribute_column="summary_vector",
    #                             content_attribute_column="movie_info_vector")
    #
    # knn_recommendations = recommender.get_recommendations()
    # print("done")

    # ------------------------------------------------------------
    # vectorized VGCF

    recommender = VGCF(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    recommender.set_column_names(user_id_column="user_id",
                                content_id_column="movie_id",
                                user_attribute_column="summary_vector",
                                content_attribute_column="movie_info_vector")

    vgcf_recommendations = recommender.get_recommendations()
    print("done")
    print(vgcf_recommendations.head())

    # ------------------------------------------------------------

    # VNMF profile
    #
    # recommender = VNMF(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    # recommender.set_column_names(user_id_column="user_id",
    #                             content_id_column="movie_id",
    #                             user_attribute_column="summary_vector",
    #                             content_attribute_column="movie_info_vector")
    #
    # recommender.train_model()
    #
    # vnmf_recommendations = recommender.get_recommendations()
    # print("done")
    # print(vnmf_recommendations.head())

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Instantiate the evaluator
    rec_evaluator = RecEvaluator(
        test_df=testset_df,
        recommendations_df=vgcf_recommendations,
        user_id_col="user_id",
        item_id_col="movie_id",
        rating_col="rating",  # Column in 'testset_df' for ratings
        rank_col="recommendation_rank",  # Column in 'recommendations' for ranking
        relevance_threshold=4.0  # Typically 4.0+ means "relevant" in MovieLens
    )

    # 4. Evaluate at several K values (commonly used in academic papers)
    k_values = [1, 5, 10, 20, 30]

    # Compute all metrics
    results_df = rec_evaluator.evaluate_all_metrics(k_values)

    # 5. Print or inspect the results
    print("Evaluation Results:\n", results_df)

    # 6. Plot the metrics and store the plot
    #    By default, it plots Precision, Recall, NDCG, MRR, MAP
    #    but we can also specify exactly which to plot:
    metrics_to_plot = ["Precision", "Recall", "NDCG", "MRR", "MAP"]
    rec_evaluator.plot_metrics(results_df=results_df, metrics=metrics_to_plot, save_path="evaluation_plot.png")

    # 7. Save the numeric results to a CSV file
    rec_evaluator.save_results_to_csv(results_df=results_df, csv_path="evaluation_results.csv")

    # Now you have:
    # - A CSV file with the numeric results
    # - A plot image (e.g., "evaluation_plot.png") showing metric curves vs. K







