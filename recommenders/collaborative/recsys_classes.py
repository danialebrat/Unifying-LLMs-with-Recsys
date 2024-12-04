from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """
    Abstract base class for a recommender system.
    """

    @abstractmethod
    def train(self, data):
        """
        Train the recommender model with the given data.
        """
        pass

    @abstractmethod
    def recommend(self, user_id, users_items, N=10):
        """
        Recommend a list of items for a given user.
        """
        pass


from scipy.sparse import csr_matrix
from typing import List, Union

class ImplicitRecommender(BaseRecommender):
    def __init__(self, model):
        """
        Initialize the ImplicitRecommender with a model and a function to retrieve the user-items matrix.
        :param model: A model from the Implicit library with fit and recommend methods.
        :param get_user_items_func: A function that returns the latest user-items csr_matrix.
        """
        self.model = model
        #self.get_user_items_func = get_user_items_func

    def train(self, users_items: csr_matrix):
        """
        Train the model. The item_users should be a csr_matrix with shape (num_items, num_users).
        """
        self.model.fit(users_items)

    def recommend(self, user_ids: Union[int, List[int]], users_items: csr_matrix, N=10):
        """
        Recommend items for given user(s) using the latest user-item matrix.
        :param user_ids: A single user ID or a list of user IDs.
        :param users_items: The user-item interaction matrix.
        :param N: Number of recommendations to return.
        """
        if isinstance(user_ids, int):
            # For a single user ID, wrap it in a list
            user_ids = [user_ids]
        return self.model.recommend(user_ids, users_items[user_ids], N=N)