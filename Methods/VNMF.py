from sklearn.decomposition import PCA, NMF

import Recommender
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------------

from scipy.sparse import lil_matrix

class VNMF(Recommender):
    """
    VNMF class that integrates textual embeddings into NMF-based recommendation.
    """

    def __init__(self, content_df, interactions_df, users_df,
                 n_components=384, max_iter=200, random_state=42, alpha=0.0, l1_ratio=0.0):
        super().__init__(content_df=content_df, interactions_df=interactions_df, users_df=users_df)
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.U = None  # User factors
        self.V = None  # Item factors
        self.user_mapping = None
        self.item_mapping = None
        self.R = None  # rating matrix
        self.known_mask = None  # mask of known interactions

        # Column names (set via set_column_names)
        self.user_id_column = None
        self.content_id_column = None
        self.user_attribute_column = None
        self.content_attribute_column = None

        self.recommendations_df = None

    def _prepare_data(self):
        """
        Prepare the interaction matrix R and mappings from IDs to matrix indices.
        """
        if (self.user_id_column is None or self.content_id_column is None):
            raise ValueError("Column names must be set before preparing data.")

        # Extract unique user and item IDs
        unique_user_ids = self.interactions_df[self.user_id_column].unique()
        unique_item_ids = self.interactions_df[self.content_id_column].unique()

        self.user_mapping = {uid: i for i, uid in enumerate(unique_user_ids)}
        self.item_mapping = {iid: i for i, iid in enumerate(unique_item_ids)}

        num_users = len(unique_user_ids)
        num_items = len(unique_item_ids)

        # Initialize rating matrix with NaNs to indicate unknown
        R = np.full((num_users, num_items), np.nan)

        # Fill in known ratings/interactions
        for _, row in self.interactions_df.iterrows():
            u_idx = self.user_mapping[row[self.user_id_column]]
            i_idx = self.item_mapping[row[self.content_id_column]]
            R[u_idx, i_idx] = row["rating"]  # assuming interactions_df has a "rating" column

        # Create a mask for known entries
        known_mask = ~np.isnan(R)

        # Replace NaN with 0 for NMF fitting, but keep mask to know which were known
        R_filled = np.nan_to_num(R, nan=0.0)

        self.R = R_filled
        self.known_mask = known_mask

    def _get_textual_embeddings(self):
        """
        Extract user and item embeddings from the provided dataframes.
        Assume the embeddings are stored as lists of floats in user_attribute_column and content_attribute_column.
        """
        # Sort dfs by user/item id to align with user_mapping/item_mapping
        # Ensuring same order of users and items as in R matrix construction
        user_df_sorted = self.users_df.set_index(self.user_id_column).loc[list(self.user_mapping.keys())]
        item_df_sorted = self.content_df.set_index(self.content_id_column).loc[list(self.item_mapping.keys())]

        user_embeddings = np.array(user_df_sorted[self.user_attribute_column].tolist())
        item_embeddings = np.array(item_df_sorted[self.content_attribute_column].tolist())

        return user_embeddings, item_embeddings

    def _align_embedding_dimensions(self, user_embeddings, item_embeddings):
        """
        Align the dimensions of the given embeddings to n_components.
        If embeddings have a different dimension than n_components,
        we can use PCA to reduce or project them.
        """
        emb_dim = user_embeddings.shape[1]

        if emb_dim > self.n_components:
            # Reduce dimension using PCA
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            user_embeddings = pca.fit_transform(user_embeddings)
            item_embeddings = pca.fit_transform(item_embeddings)
        elif emb_dim < self.n_components:
            # If fewer dimensions, pad with zeros (simple approach)
            pad_width = self.n_components - emb_dim
            user_embeddings = np.hstack([user_embeddings, np.zeros((user_embeddings.shape[0], pad_width))])
            item_embeddings = np.hstack([item_embeddings, np.zeros((item_embeddings.shape[0], pad_width))])

        return user_embeddings, item_embeddings

    def train_model(self):
        """
        Train the VNMF model:
        1. Prepare data (R matrix, mapping).
        2. Extract and align textual embeddings.
        3. Initialize U and V from textual embeddings.
        4. Fit NMF using these initializations.
        5. Optionally incorporate a custom regularization or a post-processing step.
        """
        self._prepare_data()

        user_embeddings, item_embeddings = self._get_textual_embeddings()
        user_embeddings, item_embeddings = self._align_embedding_dimensions(user_embeddings, item_embeddings)

        # Initialize NMF with textual embeddings as U and V
        # NMF in sklearn uses W (user) and H (item) ~ U and V here
        nmf_model = NMF(n_components=self.n_components, init='custom', max_iter=self.max_iter,
                        random_state=self.random_state, alpha=self.alpha, l1_ratio=self.l1_ratio)

        W_init = np.clip(user_embeddings, 1e-5, None)  # ensure no zeros for NMF
        H_init = np.clip(item_embeddings.T, 1e-5, None)  # transpose to match NMF shape

        # Fit the NMF model
        nmf_model.fit(self.R, W=W_init, H=H_init)

        self.U = nmf_model.transform(self.R)  # user factors
        self.V = nmf_model.components_.T  # item factors

        # Align learned factors closer to embeddings (post-processing)
        # Item's learned factors should be closer to embeddings, but we can give more room to user's factors as it represents user's preferences
        # A real approach would integrate this into the optimization objective.

        # Enhance with neural layers:
        # self.enhance_with_neural_layers()

    def enhance_with_neural_layers(self):
        """
        Placeholder for adding a neural network layer on top of U and V.
        This could be something like:
        - Train a neural network that takes [U(u), V(i)] and predicts rating.
        - Fine-tune U and V within that network.

        Not implemented in this example.
        """
        pass

    def get_recommendations(self, top_n=10):
        """
        Generate top-N recommendations for each user.
        """
        if self.U is None or self.V is None:
            raise ValueError("Model has not been trained yet.")

        # Predict ratings: R_pred = U * V^T
        R_pred = self.U @ self.V.T

        # For known items, we might want to exclude them from recommendations
        # Set their predicted rating to a very low number to exclude
        R_pred[self.known_mask] = -np.inf

        recommendations = []

        # Reverse mapping to get original IDs
        inv_user_mapping = {v: k for k, v in self.user_mapping.items()}
        inv_item_mapping = {v: k for k, v in self.item_mapping.items()}

        for u_idx in range(R_pred.shape[0]):
            user_id = inv_user_mapping[u_idx]
            # Get top-N item indices
            top_items = np.argpartition(R_pred[u_idx, :], -top_n)[-top_n:]
            # Sort them by predicted rating
            top_items = top_items[np.argsort(R_pred[u_idx, top_items])][::-1]

            for rank, i_idx in enumerate(top_items, 1):
                item_id = inv_item_mapping[i_idx]
                recommendations.append((user_id, item_id, rank, "VNMF"))

        self.recommendations_df = pd.DataFrame(recommendations,
                                               columns=[self.user_id_column,
                                                        self.content_id_column,
                                                        'recommendation_rank',
                                                        'module_source'])

        # Convert data types
        self.recommendations_df = self.recommendations_df.astype(
            {self.user_id_column: 'int',
             self.content_id_column: 'int',
             'recommendation_rank': 'int',
             'module_source': 'str'}
        )

        return self.recommendations_df

    # Additional utility methods can be added as needed
