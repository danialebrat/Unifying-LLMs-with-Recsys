# The Snowpark package is required for Python Worksheets.
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from datetime import datetime
import json
from snowflake.snowpark import Session
import pandas as pd
from scipy.sparse import coo_matrix
import joblib
import io
# import sys
# import os
import base64
import json


# ----------------------------------------------------------------------------------------------------
def get_user_items(session, brand, section, u_encoder, i_encoder):

    # session.sql("USE DATABASE RECSYS").collect()
    # session.sql("USE SCHEMA PREPROCESSED_TABLES").collect()

    # Specify your table name
    table_name = "RECSYS_V2.PREPROCESSED_TABLES.%s_%s_INTERACTION" % (brand, section)

    # Query the table
    snowpark_dataframe = session.table(table_name).select(
        col("USER_ID"),
        col("CONTENT_ID"),
        col("INTERACTION_COUNT")
    )

    # snowpark_dataframe = snowpark_dataframe.drop_duplicates(["USER_ID", "CONTENT_ID"])

    # Convert to Pandas DataFrame
    pandas_df = snowpark_dataframe.toPandas()

    # Ensure INTERACTION_COUNT is numeric
    pandas_df["INTERACTION_COUNT"] = pd.to_numeric(pandas_df["INTERACTION_COUNT"], errors = "coerce")

    # Convert USER_ID and CONTENT_ID to category codes
    # pandas_df["USER_ID"] = pandas_df["USER_ID"].astype("category").cat.codes
    # pandas_df["CONTENT_ID"] = pandas_df["CONTENT_ID"].astype("category").cat.codes
    print(pandas_df.head(15))
    print(pandas_df.tail())

    pandas_df["USER_ID"] = u_encoder.fit_transform(pandas_df["USER_ID"])
    pandas_df["CONTENT_ID"] = i_encoder.fit_transform(pandas_df["CONTENT_ID"])

    # Drop duplicates
    df_unique_interactions = pandas_df.drop_duplicates(subset=["USER_ID", "CONTENT_ID"])
    # df_unique_interactions = pandas_df

    # Create a sparse matrix from the unique interactions
    sparse_matrix = coo_matrix(
        (df_unique_interactions["INTERACTION_COUNT"],
                                  (df_unique_interactions["USER_ID"], df_unique_interactions["CONTENT_ID"]))).tocsr()
    print("matrix created")
    # print(sparse_matrix)
    return sparse_matrix

# ----------------------------------------------------------------------------------------------------

def store_recommendations(session, recommendations, brand, section, batch_size=100):
    table_name = f"RECSYS_V2.INITIAL_RECOMMENDATIONS.NEW_{brand}_{section}_CF_RECOMMENDATIONS"
    print("Creating table", table_name)

    # Create the recommendations table if it doesn"t exist
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            user_id INT,
            content_id INT,
            recommendation_rank INT
        )
    """
    session.sql(create_table_sql).collect()

    # Prepare and insert records in batches
    for batch in chunked_recommendations(recommendations, batch_size):
        values_to_insert = ", ".join([f"({user_id}, {item_id}, {rank})"
                                      for user_id, recs in batch.items()
                                      for rank, item_id in enumerate(recs, start=1)])
        insert_sql = f"INSERT INTO {table_name} (user_id, content_id, recommendation_rank) VALUES {values_to_insert}"
        session.sql(insert_sql).collect()


# ----------------------------------------------------------------------------------------------------

def chunked_recommendations(recommendations, batch_size):
    """Yield successive batch_size chunks of recommendations."""
    batch = {}
    for count, (user_id, recs) in enumerate(recommendations.items(), start=1):
        batch[user_id] = recs
        if count % batch_size == 0:
            yield batch
            batch = {}
    if batch:
        yield batch


# ----------------------------------------------------------------------------------------------------
def recommend_batch_implicit(session, user_ids, brand, section):
    table_name = f"RECSYS_V2.CF_MODELS.{brand}_{section}_MODELS"  # Replace with dynamic brand and section values

    # Query to retrieve the most recent model and encoders from the specific table
    query = f"SELECT * FROM {table_name} ORDER BY model_id DESC LIMIT 1"
    result = session.sql(query).collect()

    # Assuming result[0] contains the latest record
    latest_model_base64 = result[0]["MODEL_DATA"]
    latest_user_encoder_base64 = result[0]["USER_ENCODER_DATA"]
    latest_item_encoder_base64 = result[0]["ITEM_ENCODER_DATA"]

    # Function to deserialize
    def deserialize(base64_data):
        byte_data = base64.b64decode(base64_data)
        byte_stream = io.BytesIO(byte_data)
        return joblib.load(byte_stream)

    # Deserialize model and encoders
    model = deserialize(latest_model_base64)
    user_encoder = deserialize(latest_user_encoder_base64)
    item_encoder = deserialize(latest_item_encoder_base64)

    user_item_matrix = get_user_items(session, brand, section, user_encoder, item_encoder)

    # Check if user_id is a single ID or a list of IDs
    is_single_user = isinstance(user_ids, int)
    user_indices = user_encoder.transform([user_ids]) if is_single_user else user_encoder.transform(user_ids)
    # print(user_indices)

    try:
        # Generate recommendations using the ALS model
        recommendations = model.recommend(user_indices, user_item_matrix, N=100)  # Adjust N as needed

    # IndexError will occasionally happen if a user just signed up and the model hasn"t been retrained yet --
    # but should be rare as we will normally retrain models immediately prior to making batch recommendations
    except IndexError:
        # Handle the error and return immediately
        return json.dumps({})

    if is_single_user:
        # If it"s a single user, extract item IDs and scores directly
        itemids, scores = recommendations

        itemids = itemids.flatten()
        decoded_recommendations = item_encoder.inverse_transform(itemids)
        # Create a dictionary for the single user"s recommendations
        decoded_recommendations = {user_ids: decoded_recommendations.tolist()}
        # decoded_recommendations = item_encoder.inverse_transform(itemids)

        # Create a dictionary for the single user"s recommendations

        # decoded_recommendations = {user_ids: decoded_recommendations.tolist()}
    else:
        # For multiple users, process each user"s recommendations
        decoded_recommendations = {}
        for idx, user_recommendations in enumerate(zip(*recommendations)):
            itemids, scores = user_recommendations
            decoded_items = item_encoder.inverse_transform(itemids)
            # decoded_recommendations[user_indices[idx]] = decoded_items.tolist()
            decoded_recommendations[user_ids[idx]] = decoded_items.tolist()

    # Convert keys to int and values to list of ints,
    # compatible w/ json.dumps()

    converted_dict = {int(k): list(map(int, v)) for k, v in decoded_recommendations.items()}

    # write them to the appropriate recommendations database
    store_recommendations(session, converted_dict, brand, section, batch_size=100)


# ----------------------------------------------------------------------------------------------------

def generate_recs_all_users(session, brand, section):
    # Specify your table name
    table_name = 'RECSYS_V2.PREPROCESSED_TABLES.%s_%s_INTERACTION' % (brand, section)

    # Query to retrieve all unique user IDs
    user_ids_df = session.table(table_name).select(col("USER_ID")).distinct().collect()
    user_ids = [row['USER_ID'] for row in user_ids_df]

    # Function to divide users into batches
    def divide_batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Example batch size
    batch_size = 1000  # Adjust as needed
    user_batches = list(divide_batches(user_ids, batch_size))

    old_table_name = "RECSYS_V2.INITIAL_RECOMMENDATIONS.%s_%s_CF_RECOMMENDATIONS" % (brand, section)
    rec_table_name = "RECSYS_V2.INITIAL_RECOMMENDATIONS.NEW_%s_%s_CF_RECOMMENDATIONS" % (brand, section)
    back_table_name = "RECSYS_V2.INITIAL_RECOMMENDATIONS.BACKUP_%s_%s_CF_RECOMMENDATIONS" % (brand, section)

    # just the first time
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS {} (
            user_id INT,
            content_id INT,
            recommendation_rank INT
        )
        """.format(old_table_name)
    session.sql(create_table_sql).collect()

    # Create the new table
    # Create the new recommendations table
    session.sql(f"CREATE TABLE IF NOT EXISTS {rec_table_name} LIKE {old_table_name}").collect()

    print('number of batches: ', len(user_batches))
    counter = 0
    for batch in user_batches:
        counter += 1
        print('batch: ', counter)
        # Call your stored procedure for each batch
        recommend_batch_implicit(session,batch, brand, section)
        # session.call("batch_recommend_implicit", batch, brand, section)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_backup_name = f"{back_table_name}_{timestamp}"

    try:
        rename_sql = f"ALTER TABLE {back_table_name} RENAME TO {timestamped_backup_name}"
        session.sql(rename_sql).collect()
    except:
        print('No Backup Table Yet')
    # Rename the old table to backup
    session.sql(f"ALTER TABLE {old_table_name} RENAME TO {back_table_name}").collect()

    # Rename the new table to the original name
    session.sql(f"ALTER TABLE {rec_table_name} RENAME TO {old_table_name}").collect()

    print(f'All Recs Generated: {brand} _ {section}')
    return "Success"


# ----------------------------------------------------------------------------------------------------
def load_config_(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    credential_info_file = '../../Config_files/snowflake_credentials_Danial.json'
    credential_info = load_config_(credential_info_file)

    conn = {
        "user": credential_info['user'],
        "password": credential_info['password'],
        "account": credential_info['account'],
        "role": credential_info['role'],
        "database": credential_info['database'],
        "warehouse": credential_info['warehouse'],
        "schema": credential_info['schema'],
    }

    session = Session.builder.configs(conn).create()

    # brands = ['SINGEO', 'GUITAREO', 'PIANOTE', 'DRUMEO']
    brands = ['SINGEO', 'PIANOTE']
    # sections = ['SONG', 'QUICK_TIPS', 'WORKOUT', 'COURSE']
    sections = ['COURSE']

    for brand in brands:
        for section in sections:
            generate_recs_all_users(session, brand, section)

    # generate_recs_all_users(session, 'DRUMEO', 'COURSE')
