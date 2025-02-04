
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

# import Lusifer
from Lusifer import Lusifer

# Use your actual API key securely
KEY = os.getenv('OPENAI_API_KEY')


# --------------------------------------------------------------
def load_recommendations(data="100k"):
    # folder that contains csv files of recommendation results
    path = f"../output/{data}"

    # loading each csv file as pandas dataframe from below path
    recommenders_paths = {
        "VGCF": f"{path}/vgcf_recommendations_{data}.csv",
        "LightGCN": f"{path}/lightgcn_result_{data}.csv",
        "NGCF": f"{path}/ngcf_result_{data}.csv",
        "ALS": f"{path}/als_result_{data}.csv"
    }

    # Load each recommendation file into a DataFrame
    recommendations_dfs = []
    for recommender, file_path in recommenders_paths.items():
        df = pd.read_csv(file_path)
        df['recommender'] = recommender  # Add a column to identify the recommender
        recommendations_dfs.append(df)

    # Concatenate all dataframes
    recommendations_df = pd.concat(recommendations_dfs, ignore_index=True)

    # Drop duplicates where there is no more than a pair of user_id and movie_id.
    recommendations_df = recommendations_df.drop_duplicates(subset=['user_id', 'movie_id'])

    return recommendations_df

# --------------------------------------------------------------
def enrich_test_set(test_set, new_data):
    # Create a unique identifier for each pair of user_id and movie_id
    test_set['unique_pair'] = test_set['user_id'].astype(str) + '_' + test_set['movie_id'].astype(str)
    new_data['unique_pair'] = new_data['user_id'].astype(str) + '_' + new_data['movie_id'].astype(str)

    # Filter out rows in new_data that already exist in test_set
    new_data_filtered = new_data[~new_data['unique_pair'].isin(test_set['unique_pair'])]

    # Add the new records to the test_set
    enriched_test_set = pd.concat([test_set, new_data_filtered], ignore_index=True)

    # Drop the unique_pair column as it's no longer needed
    enriched_test_set = enriched_test_set.drop(columns=['unique_pair'])

    # Print how many new records we added
    print(f"Added {len(new_data_filtered)} new records to the test set.")

    return enriched_test_set
# --------------------------------------------------------------
def loading_data(data="100k"):
    user_path = f"../Dataset/{data}/users_with_summary_df.csv"
    movie_path = f"../Dataset/{data}/movies_enriched_dataset.pkl"
    trainset_path = f"../Dataset/{data}/train_test_sets/u1.base"
    testset_path = f"../Dataset/{data}/train_test_sets/u1.test"

    users_df = pd.read_csv(user_path)
    movies_df = pd.read_pickle(movie_path)
    movies_df = movies_df[['movie_id', 'movie_info']]
    train_set = pd.read_csv(trainset_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
    test_set = pd.read_csv(testset_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')

    return users_df, movies_df, train_set, test_set

# --------------------------------------------------------------


if __name__ == "__main__":

    # loading data
    users_df, movies_df, train_df, test_df = loading_data("1m")

    # loading recommendations
    recommendations_df = load_recommendations("1m")
    test_df = enrich_test_set(test_set=test_df, new_data=recommendations_df)

    # create a Lusifer object
    lusifer = Lusifer(users_df=users_df,
                      items_df=movies_df,
                      ratings_df=train_df)

    # set API connection
    # model = "gpt-4o-mini-2024-07-18"
    model = "gpt-4o-mini"
    lusifer.set_openai_connection(KEY, model=model)

    # set column names
    lusifer.set_column_names(user_feature="user_info",
                             item_feature="movie_info",
                             user_id="user_id",  # set by default
                             item_id="movie_id",
                             timestamp="timestamp",  # set by default
                             rating="rating")  # set by default

    # set LLM initial instruction
    instructions = """You are an AI assistant that receives users information and try to act like the user  by 
    analysing user's characteristics, profile and historical ratings inorder to provide new ratings for the recommended movies"""

    lusifer.set_llm_instruction(instructions)

    # Filtering out invalid movie_ids, make sure we have movie_info for every movie in the test set
    test_df = lusifer.filter_ratings(test_df)
    train_df = lusifer.filter_ratings(train_df)

    # Create a new column for simulated ratings
    test_df['simulated_ratings'] = test_df['rating']

    user_ids = test_df['user_id'].unique()
    # user_ids = [1]

    temp_token_counter = 0
    previous_token_total = 0

    # laod test_set
    # test_df = pd.read_pickle("enriched_test_set.pkl")

    ## PHASE 2: Generating Simulated ratings
    for user_id in tqdm(user_ids, desc="Generating simulated ratings"):
        # isolating user's ratings in the test set
        user_ratings = test_df[test_df['user_id'] == user_id]

        # we might have some values from previous run
        missing_ratings = user_ratings[user_ratings['simulated_ratings'].isnull()]

        # getting the summary for the user
        user_profile = users_df.loc[users_df['user_id'] == user_id, 'summary'].values[0]

        # we will not run this part if we have all ratings for the user
        if not missing_ratings.empty:
            last_N_movies = lusifer.get_last_ratings(user_id, n=5)

            # generate rating
            llm_ratings = lusifer.rate_new_items(user_profile, last_N_movies, missing_ratings)

            # Assigning the ratings to the movies
            for movie_id, rating in llm_ratings.items():
                test_df.loc[(test_df['user_id'] == user_id) & (
                        test_df['movie_id'] == movie_id), 'simulated_ratings'] = rating

            lusifer.save_data(test_df, 'enriched_test_set')

    total_prompt_tokens = lusifer.total_tokens['prompt_tokens'] + lusifer.total_tokens['completion_tokens']
    total_cost = ((lusifer.total_tokens['prompt_tokens'] / 1000000) * 0.15) + (
            (lusifer.total_tokens['completion_tokens'] / 1000000) * 0.6)  # Cost calculation estimation

    print("\nToken Usage and Cost:")
    print(f"Prompt Tokens: {lusifer.total_tokens['prompt_tokens']}")
    print(f"Completion Tokens: {lusifer.total_tokens['completion_tokens']}")
    print(f"Total Tokens: {lusifer.total_tokens['prompt_tokens'] + lusifer.total_tokens['completion_tokens']}")
    print(f"Estimated Cost (USD): {total_cost:.5f}")

    print("\nOverall Metrics:")


