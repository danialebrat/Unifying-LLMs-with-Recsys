import pandas as pd
from Recommender import ClusteredKNN, UserBasedRecommender
from VGCF import VGCF
from embedding_layer import EmbeddingLayer

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
    # vectorized NMF

    recommender = VGCF(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    recommender.set_column_names(user_id_column="user_id",
                                content_id_column="movie_id",
                                user_attribute_column="summary_vector",
                                content_attribute_column="movie_info_vector")

    vgcf_recommendations = recommender.get_recommendations()
    print("done")
    print(vgcf_recommendations.head())







