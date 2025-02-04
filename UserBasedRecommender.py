import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from Recommender import Recommender


# ----------------------------------------------------------------------------------------------------
# user_based content_recommender
class UserBasedRecommender(Recommender):

    def __init__(self, content_df, interactions_df, users_df):
        super().__init__(content_df, interactions_df, users_df)

    def get_recommendations(self):

        self.all_vectors = np.vstack(self.content_df[self.content_attribute_column].tolist())

        # calculate cosine similarity for all users
        self.calculate_cosine_similarity()

        # generate initial recommendations based on cosine similarity profiles
        self.generate_recommendations()

        return self.recommendations_df

    # ----------------------------------------------------------------------------------------------------
    def calculate_cosine_similarity(self, batch_size=50):
        """
        Calculate cosine similarity in batches to manage memory usage.
        :return: None (update user_profiles and cosine_scores)
        """
        user_features = self.user_attribute_column
        num_users = len(self.users_df)

        for start_idx in tqdm(range(0, num_users, batch_size), desc="Batch Processing Users"):
            end_idx = min(start_idx + batch_size, num_users)
            user_matrix = np.vstack(self.users_df.iloc[start_idx:end_idx][user_features].tolist())
            batch_scores = cosine_similarity(user_matrix, self.all_vectors)

            for idx, scores in enumerate(batch_scores):
                user_id = self.users_df.iloc[start_idx + idx][self.user_id_column]
                self.cosine_scores[user_id] = scores

    # ----------------------------------------------------------------------------------------------------
    def generate_recommendations(self):
        """
        Generate recommendations ensuring diversity.
        :return: recommendations dataframe
        """
        recommendations = []

        for user_id, scores in tqdm(self.cosine_scores.items(), desc="Generating Recommendations"):
            indices = np.argsort(scores)[-len(scores):][::-1]
            recommended_contents = self.content_df.iloc[indices]

            # Collect recommendations considering artist diversity
            user_recommendations = []
            for _, row in recommended_contents.iterrows():
                if len(user_recommendations) >= self.total_recommendations:
                    break
                else:
                    user_recommendations.append((user_id, row[self.content_id_column], len(user_recommendations) + 1))

            recommendations.extend(user_recommendations)

        # Create DataFrame from a recommendation list
        self.recommendations_df = pd.DataFrame(recommendations,
                                               columns=[self.user_id_column, self.content_id_column,
                                                        'recommendation_rank'])

        # adding source of the recommendations
        row_numbers = self.recommendations_df.shape[0]  # Gives number of rows
        module_source_value = "user_profile"
        module_source = [module_source_value] * row_numbers
        self.recommendations_df['module_source'] = module_source

        # Convert data types of the columns to integers
        self.recommendations_df = self.recommendations_df.astype(
            {self.user_id_column: 'int', self.content_id_column: 'int', 'recommendation_rank': 'int',
             'module_source': 'str'})

        return self.recommendations_df

# How to use the class:

# UserBasedRecommender(content_df, interactions_df, users_df)
# set_column_names(user_id_column, content_id_column, user_attribute_column, content_attribute_column)
# recommendations_df = get_recommendations()
