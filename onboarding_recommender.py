"""
This module will generate K recommendations using content-based filtering method for each brand/section using
user's onboarding answers.
we will store the recommendations as initial recommendations which we would filter them later using post filter module.
This module contains the below steps:
- reading preprocessed content table and interaction table. (content_df and interaction_df)
- reading onboarding answers for the requested brand.
- Converting JSON arrays to numpy vectors
- generating user profiles:
- generating recommendations using the K nearest neighbor for each user based on the feature associated to the section.
- making sure to have 45 recommendation per user and sorting the recommendations based on the score
- Adding popular Contents at the end of the  list.
- uploading the initial recommendations back to snowflake

"""

# ----------------------------------------------------------------------------------------------------
# Importing necessary libraries

import numpy as np
import pytz
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json
from abc import ABC, abstractmethod
import ast
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session


# ----------------------------------------------------------------------------------------------------
# Define the DataLoader class
class ContentBasedRecommender(ABC):
    """
    This is an abstract class for content based recommender systems. specific methods will be subclasses.
    on the recommendations.
    """

    def __init__(self, brand, section, destination_paths, content_preprocessing_info, session):
        self.brand = brand
        self.section = section
        self.session = session

        # config files
        self.destination_paths = destination_paths
        self.content_preprocessing_info = content_preprocessing_info

        # dataframes
        self.content_df = None  # current content_df on the snowflake
        self.interactions_df = None  # current interactions_df on the snowflake
        self.popular_df = None  # current popular_df on the snowflake
        self.user_profile_df = None  # current preprocessed user_profile_df on snowflake
        self.recommendations_df = None
        self.permission_df = None  # current permission table

        # storing necessary files
        self.cosine_scores = {}
        self.all_vectors = None

        self.total_recommendations = 45

    # ----------------------------------------------------------------------------------------------------
    def run_read_query(self, query, data_type):
        """
        Executes a SQL query on Snowflake that fetches the data from initial tables
        :return: Snowflake dataframe containing the query results
        """
        try:
            # Execute the query and fetch results as a DataFrame
            dataframe = self.session.sql(query).to_pandas()
            print(f"{self.brand} _ {self.section} _ {data_type} : reading data from initial table successfully")
            return dataframe
        except Exception as e:
            print(f"Error in reading data from table: {e}")

    # ----------------------------------------------------------------------------------------------------
    def run_write_query(self, query, dataframe, method="CB"):
        """
        Executes a SQL query on Snowflake that writes the recommendations tables after post-filtering
        :param query: SQL query string to be executed
        :param dataframe: finalize recommendation dataframe
        :return: None
        """
        # Resetting the DataFrame index to ensure it's a RangeIndex
        dataframe = dataframe.reset_index(drop=True)
        try:
            database = self.destination_paths['initial_recommendations']['database']
            schema = self.destination_paths['initial_recommendations']['schema']
            self.session.use_database(database)
            self.session.use_schema(schema)
            # Create or replace the table using the Snowflake session
            self.session.sql(query).collect()
            print(f"Table {self.brand}_{self.section}_{method}_RECOMMENDATIONS created/updated successfully.")

            # Replace any hyphens in self.section with underscores
            safe_section = self.section.replace("-", "_")

            # Construct the full table name with safe_section
            table_name = f"{self.brand}_{safe_section}_{method}_RECOMMENDATIONS".upper()
            table_name = table_name.replace('"', '')

            # Assuming write_pandas is compatible with Snowflake session
            self.session.write_pandas(dataframe, table_name.strip(), auto_create_table=True)

            print(f"Data inserted into {table_name} successfully.")
        except Exception as e:
            print(f"Error in creating/updating/inserting table: {e}")

    # ----------------------------------------------------------------------------------------------------
    def generate_read_sql_query(self, data_type):
        """
        Generates an SQL query to read the data from preprocessed tables.
        :return: a string contains query
        """
        # Replace any hyphens in self.section with underscores
        global initial_table_path
        safe_section = self.section.replace("-", "_")
        if data_type == "popular":
            data_type = "POPULAR_ITEMS_RECOMMENDATIONS"
            table_name = f"{self.brand}_{safe_section}_{data_type}".upper()
            initial_table_path = f"{self.destination_paths['initial_recommendations']['destination_path']}.{table_name}".upper()

        elif data_type == "permission":
            table_name = f"user_{data_type}".upper()
            initial_table_path = (
                f"{self.destination_paths['permission']['destination_path']}.{table_name}"
                .upper())

        elif data_type == "user_profile":
            table_name = f"{self.brand}_USERS".upper()
            initial_table_path = f"{self.destination_paths['preprocessed_tables']['destination_path']}.{table_name}".upper()

        elif data_type == "content":
            table_name = f"{self.brand}_{safe_section}_{data_type}".upper()
            initial_table_path = f"{self.destination_paths['preprocessed_tables']['destination_path']}.{table_name}".upper()

        elif data_type == "interaction":
            table_name = f"{self.brand}_{safe_section}_{data_type}".upper()
            initial_table_path = f"{self.destination_paths['preprocessed_tables']['destination_path']}.{table_name}".upper()

        else:
            print(f"Not valid data type : {data_type}")
            initial_table_path = ""

        query = f"SELECT * FROM {initial_table_path}"

        return query

    # ----------------------------------------------------------------------------------------------------
    def generate_write_sql_query(self, dataframe, method="CB"):
        """
        Generates an SQL query to create/update initial content-based recommendation.
        :return: a string contains query
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a DataFrame as input")

        # If any column is inadvertently a Series, convert it back to DataFrame
        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()

        # Replace any hyphens in self.section with underscores
        safe_section = self.section.replace("-", "_")

        # Construct the full table name with safe_section
        table_name = f"{self.brand}_{safe_section}_{method}_RECOMMENDATIONS"
        full_table_path = f"{self.destination_paths['initial_recommendations']['destination_path']}.{table_name}"

        # Creating the CREATE TABLE query
        create_query = f"CREATE OR REPLACE TABLE {full_table_path} ("

        # Add columns and their data types
        columns = []
        for col, dtype in dataframe.dtypes.items():
            # Use VARIANT for embedding vectors, lists, or JSON-like structures
            if isinstance(dataframe[col].iloc[0], (list, np.ndarray)):
                snowflake_type = 'VARIANT'
            elif (dtype == 'int64') | (dtype == 'int32') | (dtype == 'int8'):
                snowflake_type = 'INTEGER'
            elif dtype == 'float32':
                snowflake_type = 'FLOAT'
            elif dtype == 'bool':
                snowflake_type = 'BOOLEAN'
            else:
                snowflake_type = 'TEXT'  # Default type for other data
            # Additional data type mappings as needed

            columns.append(f"{col} {snowflake_type}")

        create_query += ', '.join(columns)
        create_query += ");"

        return create_query

    # ----------------------------------------------------------------------------------------------------
    def read_tables(self):
        """
        This function creates and runs queries to read the following tables:
        - preprocessed interaction table
        - preprocessed content table
        It then stores them as dataframes in the class.
        """
        data_types = {
            "interaction": "interactions_df",
            "content": "content_df",
            "popular": "popular_df",
            "user_profile": "user_profile_df",
            "permission": "permission_df"
        }

        for key, data_type in data_types.items():
            # Generate the SQL query
            query = self.generate_read_sql_query(data_type=key)

            # Execute the query and store the result in the corresponding class attribute
            dataframe = self.run_read_query(query=query,
                                            data_type=key)
            # Dynamically setting the attribute based on the key
            setattr(self, data_type, dataframe)

    # ----------------------------------------------------------------------------------------------------
    @abstractmethod
    def get_recommendations(self):
        pass


# ----------------------------------------------------------------------------------------------------

class ClusteredKNN(ContentBasedRecommender):

    def get_recommendations(self):
        """
        This is the main function of the method that calls other functions in order to generate pre-computed
        recommendations for each user
        :return:None (create/update initial recommendation tables)
        """

        # reading interaction and content tables to create dataframes
        self.read_tables()

        # preprocessing dataframes
        self.preprocessing()

        # ignoring users with no valid permission
        self.filter_by_permission()

        # calculate cosine similarity for all users
        self.calculate_cosine_similarity()

        # generate initial recommendations based on cosine similarity profiles
        self.generate_recommendations()

        # adding popular content in the bottom
        self.add_popular_contents()

        # generate the writing query based on recommendation_df
        query = self.generate_write_sql_query(dataframe=self.recommendations_df, method="ON")

        # running the query to create/update the table
        self.run_write_query(query=query, dataframe=self.recommendations_df, method="ON")

    # ----------------------------------------------------------------------------------------------------
    def preprocessing(self):
        """
        Convert columns with JSON-like structures or strings to numpy arrays.
        :return: None (updating dataframe)
        """
        user_features = self.content_preprocessing_info["content_type_info"][self.section]["onboarding_feature"]
        content_feature = self.content_preprocessing_info["content_type_info"][self.section]["content_feature"]

        # Convert JSON-like structures or strings to numpy arrays
        self.content_df[content_feature] = self.content_df[content_feature].apply(self.convert_to_array)
        self.user_profile_df[user_features] = self.user_profile_df[user_features].apply(self.convert_to_array)

        # Dropping null values (WE don't have an answer from the user)
        self.user_profile_df.dropna(subset=[user_features], inplace=True)

        # storing content_vectors
        self.all_vectors = np.vstack(self.content_df[content_feature].tolist())

    # ----------------------------------------------------------------------------------------------------
    def calculate_cosine_similarity(self, batch_size=50):
        """
        Calculate cosine similarity in batches to manage memory usage.
        :return: None (update user_profiles and cosine_scores)
        """
        user_features = self.content_preprocessing_info["content_type_info"][self.section]["onboarding_feature"]
        num_users = len(self.user_profile_df)

        for start_idx in tqdm(range(0, num_users, batch_size), desc="Batch Processing Users"):
            end_idx = min(start_idx + batch_size, num_users)
            user_matrix = np.vstack(self.user_profile_df.iloc[start_idx:end_idx][user_features].tolist())
            batch_scores = cosine_similarity(user_matrix, self.all_vectors)

            for idx, scores in enumerate(batch_scores):
                user_id = self.user_profile_df.iloc[start_idx + idx]['USER_ID']
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
            seen_artists = set()

            # Collect recommendations considering artist diversity
            user_recommendations = []
            for _, row in recommended_contents.iterrows():
                if len(user_recommendations) >= self.total_recommendations:
                    break
                if row['ARTIST'] not in seen_artists or self.section != "SONG":
                    seen_artists.add(row['ARTIST'])
                    user_recommendations.append((user_id, row['CONTENT_ID'], len(user_recommendations) + 1))

            recommendations.extend(user_recommendations)

        # Create DataFrame from a recommendation list
        self.recommendations_df = pd.DataFrame(recommendations,
                                               columns=['USER_ID', 'CONTENT_ID', 'RECOMMENDATION_RANK'])

        # adding source of the recommendations
        row_numbers = self.recommendations_df.shape[0]  # Gives number of rows
        module_source_value = "onboarding"
        module_source = [module_source_value] * row_numbers
        self.recommendations_df['MODULE_SOURCE'] = module_source

        # Convert data types of the columns to integers
        self.recommendations_df = self.recommendations_df.astype(
            {'USER_ID': 'int', 'CONTENT_ID': 'int', 'RECOMMENDATION_RANK': 'int', 'MODULE_SOURCE': 'str'})

