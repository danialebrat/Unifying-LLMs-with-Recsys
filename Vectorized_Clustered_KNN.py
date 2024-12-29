import Recommender
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import heapq  # for using a heap (priority queue)

# ----------------------------------------------------------------------------------------------------

class ClusteredKNN(Recommender):

    def get_recommendations(self):
        """
        This is the main function of the method that calls other functions in order to generate pre-computed
        recommendations for each user
        :return:None (create/update initial recommendation tables)
        """

        # stacking all the vectors
        self.all_vectors = np.vstack(self.content_df[self.content_attribute_column].tolist())

        # creating using_profiles
        self.generate_user_profiles()

        # generate initial recommendations
        self.generate_recommendations()

        return self.recommendations_df

    # ----------------------------------------------------------------------------------------------------

    def generate_recommendations(self):
        """
        Optimized function for generating recommendations based on the K nearest neighbor of each cluster
        based on user previous interactions.
        """
        total_recommendations = 45
        all_recommendations = []

        # Precompute cosine similarity rankings for all users
        cosine_top_indices = {user_id: np.argsort(self.cosine_scores[user_id])[::-1][:2000] for user_id in
                              self.user_profiles}

        for user_id, top_indices in tqdm(cosine_top_indices.items(), desc="Generating recommendation for users ..."):
            user_recommendations = []
            profile = self.user_profiles[user_id]
            user_content, user_clusters = profile['content'], profile['clusters']
            user_previous_interactions = set(profile['previous_interactions'])

            # Use a set for artist tracking to avoid duplicate artist recommendations
            recommended_artists = set()

            if len(user_content[self.content_id_column].unique()) < 5:
                # Process users with few interactions as a single cluster
                for idx in top_indices:
                    if len(user_recommendations) >= total_recommendations:
                        break
                    content_id = self.content_df.iloc[idx][self.content_id_column]
                    if content_id not in user_previous_interactions:
                        recommendation_score = self.cosine_scores[user_id][idx]
                        user_recommendations.append((user_id, content_id, recommendation_score))
            else:
                cluster_counts = pd.Series(user_clusters).value_counts(normalize=True)
                for cluster, proportion in cluster_counts.items():
                    if cluster == -1:
                        continue

                    cluster_recommendation_count = max(1, int(round(total_recommendations * proportion)))
                    cluster_vector = np.mean(
                        np.vstack(user_content[user_clusters == cluster][self.content_attribute_column].tolist()),
                        axis=0)

                    # Inside your loop where you initialize NearestNeighbors
                    n_neighbors = min(cluster_recommendation_count, len(self.all_vectors))

                    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute').fit(
                        self.all_vectors)

                    distances, indices = knn.kneighbors([cluster_vector])

                    # we don't want the same contents
                    for dist, idx in zip(distances[0], indices[0]):
                        content_id = self.content_df.iloc[idx][self.content_id_column]
                        if content_id not in user_previous_interactions:
                            recommendation_score = 1 - dist
                            user_recommendations.append((user_id, content_id, recommendation_score))
                            if len(user_recommendations) >= total_recommendations:
                                break

            # Supplement recommendations if not enough
            for idx in top_indices:
                if len(user_recommendations) >= total_recommendations:
                    break
                content_id = self.content_df.iloc[idx][self.content_id_column]
                if content_id not in user_previous_interactions:
                    recommendation_score = self.cosine_scores[user_id][idx]
                    user_recommendations.append((user_id, content_id, recommendation_score))

            # Use a heap to maintain the top N recommendations
            heapq.heapify(user_recommendations)
            top_recommendations = heapq.nlargest(total_recommendations, user_recommendations, key=lambda x: x[2])

            # Convert scores to ranks
            ranked_recommendations = [(uid, cid, rank) for rank, (uid, cid, _) in
                                      enumerate(sorted(top_recommendations, key=lambda x: x[2], reverse=True), start=1)]
            all_recommendations.extend(ranked_recommendations)

        # Convert all recommendations to DataFrame and adjust types
        self.recommendations_df = pd.DataFrame(all_recommendations,
                                               columns=[self.user_id_column, self.content_id_column,
                                                        'recommendation_rank'])

        # adding source of the recommendations
        row_numbers = self.recommendations_df.shape[0]  # Gives number of rows
        module_source_value = "content_based"
        module_source = [module_source_value] * row_numbers
        self.recommendations_df['module_source'] = module_source

        # Convert data types of the columns to integers
        self.recommendations_df = self.recommendations_df.astype(
            {self.user_id_column: 'int', self.content_id_column: 'int', 'recommendation_rank': 'int',
             'module_source': 'str'})

    # ----------------------------------------------------------------------------------------------------

    def generate_user_profiles(self, batch_size=20):
        """
        Create user profiles with clusters, and precompute user vectors and previous interactions in batches.

        Args:
            batch_size: Number of users to process in each batch (default: 35)

        :return: None (update user_profiles and cosine_scores)
        """

        user_ids = self.interactions_df[self.user_id_column].unique()
        remaining_users = len(user_ids)

        with tqdm(total=len(user_ids), desc="Processing user data (batch)") as pbar:  # Use tqdm with total users
            while remaining_users > 0:
                batch_size = min(batch_size, remaining_users)  # Use minimum of remaining users and desired batch size
                user_id_chunk = user_ids[:batch_size]
                user_ids = user_ids[batch_size:]
                remaining_users -= batch_size
                pbar.update(batch_size)  # Update progress bar for each batch

                # Filter interactions and content for the current batch
                batch_interactions = self.interactions_df[self.interactions_df[self.user_id_column].isin(user_id_chunk)]
                batch_content = self.content_df[
                    self.content_df[self.content_id_column].isin(batch_interactions[self.content_id_column])]

                # Process users in the current batch
                for user_id, interactions in batch_interactions.groupby(self.user_id_column):
                    user_content = batch_content[
                        batch_content[self.content_id_column].isin(interactions[self.content_id_column])]
                    user_vectors = user_content[self.content_attribute_column].values.tolist()

                    if len(user_content) < 5:
                        user_clusters = list(range(len(user_content)))

                    else:
                        # Compute cosine distance matrix
                        distance_matrix = cosine_distances(user_vectors)
                        clusterer = HDBSCAN(min_cluster_size=5, min_samples=3, metric='precomputed', algorithm='auto',
                                            cluster_selection_method='eom')
                        user_clusters = clusterer.fit_predict(distance_matrix)

                    user_vector = np.mean(np.array(user_vectors), axis=0)
                    self.cosine_scores[user_id] = cosine_similarity([user_vector], self.all_vectors)[0]

                    # Store in user profiles (convert cluster labels to list)
                    self.user_profiles[user_id] = {
                        'content': user_content,
                        'clusters': user_clusters,
                        'previous_interactions': set(interactions[self.content_id_column].unique())
                    }


# How to use the class:

# ClusteredKNN(content_df, interactions_df, users_df)
# set_column_names(user_id_column, content_id_column, user_attribute_column, content_attribute_column)
# recommendations_df = get_recommendations()
