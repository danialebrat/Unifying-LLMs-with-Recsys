"""
Explanation
"""
import gc

# ----------------------------------------------------------------------------------------------------
# Importing necessary libraries

import numpy as np
import pytz
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from abc import ABC, abstractmethod
import heapq  # for using a heap (priority queue)
from GATRecommender import GATRecommender


# ----------------------------------------------------------------------------------------------------
# Define the DataLoader class
class Recommender(ABC):
    """
    This is an abstract class different recommendation methods
    """

    def __init__(self, content_df, interactions_df, users_df):
        # dataframes
        self.content_df = content_df
        self.interactions_df = interactions_df
        self.users_df = users_df

        # dataframe containing recommendations
        self.recommendations_df = None

        self.content_dict = None  # content_id and feature vector dictionary
        self.all_reduced_vectors = None  # for visualization purposes

        # necessary information which we calculate in the class
        self.user_profiles = {}
        self.cosine_scores = {}
        self.all_vectors = None

        self.total_recommendations = 45

        # column names: will be set by user
        self.user_id_column = None
        self.content_id_column = None
        self.user_attribute_column = None
        self.content_attribute_column = None

    def set_column_names(self, user_id_column, content_id_column, user_attribute_column, content_attribute_column):
        self.user_id_column = user_id_column
        self.content_id_column = content_id_column
        self.user_attribute_column = user_attribute_column
        self.content_attribute_column = content_attribute_column

    @abstractmethod
    def get_recommendations(self):
        pass


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


# ----------------------------------------------------------------------------------------------------
# user_based content_recommender
class UserBasedRecommender(Recommender):

    def get_recommendations(self):

        self.all_vectors = np.vstack(self.content_df[self.content_attribute_column].tolist())

        # calculate cosine similarity for all users
        self.calculate_cosine_similarity()

        # generate initial recommendations based on cosine similarity profiles
        self.generate_recommendations()

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

# ----------------------------------------------------------------------------------------------------

from scipy.sparse import lil_matrix

class VNMF(Recommender):
    def get_recommendations(self):
        # Step 1: Build user-item interaction matrix
        # Map user ids and item ids to indices
        user_ids = self.users_df[self.user_id_column].unique()
        item_ids = self.content_df[self.content_id_column].unique()
        user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
        item_id_to_index = {iid: idx for idx, iid in enumerate(item_ids)}
        index_to_user_id = {idx: uid for uid, idx in user_id_to_index.items()}
        index_to_item_id = {idx: iid for iid, idx in item_id_to_index.items()}

        num_users = len(user_ids)
        num_items = len(item_ids)

        # Build the interaction matrix R
        interactions = self.interactions_df[[self.user_id_column, self.content_id_column, 'rating']].copy()
        interactions['user_index'] = interactions[self.user_id_column].map(user_id_to_index)
        interactions['item_index'] = interactions[self.content_id_column].map(item_id_to_index)

        R = lil_matrix((num_users, num_items))
        for _, row in interactions.iterrows():
            R[row['user_index'], row['item_index']] = row['rating']

        # Step 2: Get embeddings and initialize U and V
        # User embeddings
        user_embeddings_df = self.users_df[[self.user_id_column, self.user_attribute_column]].copy()
        user_embeddings_df['user_index'] = user_embeddings_df[self.user_id_column].map(user_id_to_index)
        user_embeddings_df = user_embeddings_df.dropna(subset=[self.user_attribute_column])
        embedding_dim = len(user_embeddings_df.iloc[0][self.user_attribute_column])
        U = np.zeros((num_users, embedding_dim))
        for _, row in user_embeddings_df.iterrows():
            U[row['user_index']] = np.array(row[self.user_attribute_column])

        # Item embeddings
        item_embeddings_df = self.content_df[[self.content_id_column, self.content_attribute_column]].copy()
        item_embeddings_df['item_index'] = item_embeddings_df[self.content_id_column].map(item_id_to_index)
        item_embeddings_df = item_embeddings_df.dropna(subset=[self.content_attribute_column])
        V = np.zeros((num_items, embedding_dim))
        for _, row in item_embeddings_df.iterrows():
            V[row['item_index']] = np.array(row[self.content_attribute_column])

        # Step 3: Implement matrix factorization with SGD
        # Parameters
        num_epochs = 10
        alpha = 0.01  # Learning rate
        lambda_reg = 0.1  # Regularization parameter

        # Get list of observed ratings
        training_data = interactions[['user_index', 'item_index', 'rating']].values

        for epoch in range(num_epochs):
            np.random.shuffle(training_data)
            total_loss = 0
            for user_index, item_index, rating in training_data:
                user_index = int(user_index)
                item_index = int(item_index)
                prediction = np.dot(U[user_index], V[item_index])
                error = rating - prediction
                # Update latent factors
                U[user_index] += alpha * (error * V[item_index] - lambda_reg * U[user_index])
                V[item_index] += alpha * (error * U[user_index] - lambda_reg * V[item_index])
                total_loss += error ** 2
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}')

        # Step 4: Generate recommendations
        # For each user, compute the predicted ratings for all items
        predicted_ratings = np.dot(U, V.T)

        # For each user, get the top N recommendations
        total_recommendations = self.total_recommendations
        recommendations = []
        for user_index in range(num_users):
            user_id = index_to_user_id[user_index]
            # Exclude items the user has already interacted with
            interacted_items = interactions[interactions['user_index'] == user_index]['item_index'].tolist()
            user_predicted_ratings = predicted_ratings[user_index]
            # Set the ratings of interacted items to -inf to exclude them
            user_predicted_ratings[interacted_items] = -np.inf
            # Get top N items
            top_items_indices = np.argpartition(-user_predicted_ratings, total_recommendations)[:total_recommendations]
            top_items_scores = user_predicted_ratings[top_items_indices]
            # Sort the top items
            sorted_top_items_indices = top_items_indices[np.argsort(-top_items_scores)]
            for rank, item_index in enumerate(sorted_top_items_indices):
                item_id = index_to_item_id[item_index]
                recommendations.append({
                    self.user_id_column: user_id,
                    self.content_id_column: item_id,
                    'recommendation_rank': rank + 1
                })

        # Create the recommendations DataFrame
        self.recommendations_df = pd.DataFrame(recommendations)

        # Add module_source column
        module_source_value = "enhanced_nmf"
        row_numbers = self.recommendations_df.shape[0]
        module_source = [module_source_value] * row_numbers
        self.recommendations_df['module_source'] = module_source

        # Convert data types of the columns
        self.recommendations_df = self.recommendations_df.astype({
            self.user_id_column: 'int',
            self.content_id_column: 'int',
            'recommendation_rank': 'int',
            'module_source': 'str'
        })

# ----------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class VGCF(Recommender):
    def __init__(self, content_df, interactions_df, users_df, device=None):
        super().__init__(content_df, interactions_df, users_df)
        # Mappings from IDs to indices
        self.user_mapping = None
        self.item_mapping = None
        # PyTorch Geometric data object
        self.graph_data = None
        # GNN model
        self.model = None
        # Device (CPU or GPU)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_graph(self):
        # Build user and item ID mappings
        user_ids = self.users_df[self.user_id_column].unique()
        item_ids = self.content_df[self.content_id_column].unique()

        self.user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_mapping = {item_id: idx + len(user_ids) for idx, item_id in enumerate(item_ids)}

        # Build edge index using vectorized operations
        user_indices = self.interactions_df[self.user_id_column].map(self.user_mapping)
        item_indices = self.interactions_df[self.content_id_column].map(self.item_mapping)

        # Remove any interactions where the mapping failed (i.e., NaNs in user_indices or item_indices)
        valid_indices = user_indices.notna() & item_indices.notna()
        user_indices = user_indices[valid_indices].astype(int)
        item_indices = item_indices[valid_indices].astype(int)

        edge_index = np.vstack((np.concatenate([user_indices, item_indices]),
                                np.concatenate([item_indices, user_indices])))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Build node features
        user_embeddings = self._get_embeddings(self.users_df, self.user_id_column, self.user_attribute_column, user_ids)
        item_embeddings = self._get_embeddings(self.content_df, self.content_id_column, self.content_attribute_column,
                                               item_ids)
        node_features = torch.cat([user_embeddings, item_embeddings], dim=0)

        # Create PyTorch Geometric data object
        self.graph_data = Data(x=node_features, edge_index=edge_index).to(self.device)

    def _get_embeddings(self, df, id_column, attribute_column, ids):
        """
        Helper function to extract embeddings in the correct order.
        """
        embeddings_df = df[[id_column, attribute_column]].drop_duplicates(subset=id_column)
        embeddings_df = embeddings_df.set_index(id_column)
        embeddings_df = embeddings_df.reindex(ids)
        embeddings = torch.tensor(np.vstack(embeddings_df[attribute_column].values), dtype=torch.float)
        return embeddings

    def build_model(self):
        # Define the GAT model
        input_dim = self.graph_data.num_node_features
        hidden_dim = 64  # Hidden dimension size

        self.model = GATRecommender(input_dim, hidden_dim).to(self.device)

    def sample_negative_edges(self, num_negatives=1):
        # Map interactions to indices
        user_indices = self.interactions_df[self.user_id_column].map(self.user_mapping)
        item_indices = self.interactions_df[self.content_id_column].map(self.item_mapping)

        # Remove NaNs
        valid_indices = user_indices.notna() & item_indices.notna()
        user_indices = user_indices[valid_indices].astype(int)
        item_indices = item_indices[valid_indices].astype(int)

        positive_edges = list(zip(user_indices, item_indices))
        positive_edge_set = set(positive_edges)

        negative_edges = []
        np.random.seed(42)

        # For each user index
        unique_user_indices = user_indices.unique()
        for user_idx in unique_user_indices:
            # Get items the user has interacted with
            interacted_items = set(item_indices[user_indices == user_idx])
            # Get available items
            available_items = set(self.item_mapping.values()) - interacted_items
            available_items = list(available_items)
            if len(available_items) >= num_negatives:
                neg_items = np.random.choice(available_items, size=num_negatives, replace=False)
            else:
                neg_items = available_items
            negative_edges.extend([(user_idx, neg_item) for neg_item in neg_items])

        return positive_edges, negative_edges

    def train_model(self, epochs=75, lr=0.001):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Build mapping from user_idx to items they've interacted with
        user_indices = self.interactions_df[self.user_id_column].map(self.user_mapping)
        item_indices = self.interactions_df[self.content_id_column].map(self.item_mapping)
        valid_indices = user_indices.notna() & item_indices.notna()
        user_indices = user_indices[valid_indices].astype(int).values
        item_indices = item_indices[valid_indices].astype(int).values

        user_pos_items = {}
        for u_idx, i_idx in zip(user_indices, item_indices):
            user_pos_items.setdefault(u_idx, set()).add(i_idx)

        all_item_indices = np.array(list(self.item_mapping.values()))

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.graph_data)
            user_embeddings = out

            # Positive samples
            pos_user_indices = torch.tensor(user_indices, dtype=torch.long, device=self.device)
            pos_item_indices = torch.tensor(item_indices, dtype=torch.long, device=self.device)
            pos_user_emb = user_embeddings[pos_user_indices]
            pos_item_emb = user_embeddings[pos_item_indices]
            pos_scores = (pos_user_emb * pos_item_emb).sum(dim=1)

            # Negative samples
            neg_item_indices = []
            for u_idx in user_indices:
                u_neg_items = np.setdiff1d(all_item_indices, list(user_pos_items[u_idx]))
                if len(u_neg_items) > 0:
                    neg_item = np.random.choice(u_neg_items)
                else:
                    # If user has interacted with all items, randomly select an item
                    neg_item = np.random.choice(all_item_indices)
                neg_item_indices.append(neg_item)
            neg_user_indices = pos_user_indices
            neg_item_indices = torch.tensor(neg_item_indices, dtype=torch.long, device=self.device)
            neg_user_emb = user_embeddings[neg_user_indices]
            neg_item_emb = user_embeddings[neg_item_indices]
            neg_scores = (neg_user_emb * neg_item_emb).sum(dim=1)

            # Compute BPR loss
            loss = self.bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def bpr_loss(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    def get_recommendations(self, N=10):
        # Build graph and model if not already done
        if self.graph_data is None:
            self.build_graph()
        if self.model is None:
            self.build_model()
            self.train_model()

        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(self.graph_data).cpu()

        recommendations_list = []
        num_users = len(self.user_mapping)
        num_items = len(self.item_mapping)
        user_indices = list(self.user_mapping.values())
        item_indices = list(self.item_mapping.values())

        user_embeddings = node_embeddings[user_indices]
        item_embeddings = node_embeddings[item_indices]

        # Compute scores
        scores = torch.matmul(user_embeddings, item_embeddings.t()).numpy()  # Shape: [num_users, num_items]

        # Map indices back to IDs
        idx_to_user_id = {idx: user_id for user_id, idx in self.user_mapping.items()}
        idx_to_item_id = {idx: item_id for item_id, idx in self.item_mapping.items()}

        for i, user_idx in enumerate(user_indices):
            user_id = idx_to_user_id[user_idx]
            user_scores = scores[i]

            # Exclude items already interacted with
            interacted_items = self.interactions_df[self.interactions_df[self.user_id_column] == user_id][
                self.content_id_column].unique()
            interacted_item_indices = [self.item_mapping[item_id] - num_users for item_id in interacted_items if
                                       item_id in self.item_mapping]
            user_scores[interacted_item_indices] = -np.inf  # Exclude interacted items

            # Get top N item indices
            top_item_indices = np.argsort(-user_scores)[:N]
            recommended_item_ids = [idx_to_item_id[idx + num_users] for idx in top_item_indices]

            # Build the recommendation list with ranks and module source
            for rank, item_id in enumerate(recommended_item_ids):
                recommendations_list.append({
                    self.user_id_column: user_id,
                    self.content_id_column: item_id,
                    'recommendation_rank': rank + 1,  # Rank starts from 1
                    'module_source': 'VGCF'
                })

        # Create DataFrame
        self.recommendations_df = pd.DataFrame(recommendations_list)

        # Convert data types of the columns to appropriate types
        self.recommendations_df = self.recommendations_df.astype(
            {self.user_id_column: 'int', self.content_id_column: 'int', 'recommendation_rank': 'int'}
        )

        return self.recommendations_df
