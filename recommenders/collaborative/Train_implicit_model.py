import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from recsys_classes import ImplicitRecommender
from implicit.als import AlternatingLeastSquares
from snowflake.snowpark import Session
import json


import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from sklearn.preprocessing import LabelEncoder

import pickle
import joblib
import io
import base64
import gzip
import sys

import pandas as pd


# get data from the preprocessed tables
def get_user_items(session, brand, section, u_encoder, i_encoder):
    # session.sql("USE DATABASE RECSYS").collect()
    # session.sql("USE SCHEMA PREPROCESSED_TABLES").collect()

    # Specify your table name
    table_name = 'RECSYS_V2.PREPROCESSED_TABLES.%s_%s_INTERACTION' % (brand, section)

    # Query the table
    snowpark_dataframe = session.table(table_name).select(
        col("USER_ID"),
        col("CONTENT_ID"),
        col("INTERACTION_COUNT")
    )

    # Convert to Pandas DataFrame
    pandas_df = snowpark_dataframe.toPandas()

    # Ensure INTERACTION_COUNT is numeric
    pandas_df['INTERACTION_COUNT'] = pd.to_numeric(pandas_df['INTERACTION_COUNT'], errors='coerce')

    # Convert USER_ID and CONTENT_ID to category codes
    # pandas_df['USER_ID'] = pandas_df['USER_ID'].astype("category").cat.codes
    # pandas_df['CONTENT_ID'] = pandas_df['CONTENT_ID'].astype("category").cat.codes
    print(pandas_df.head())

    pandas_df['USER_ID'] = u_encoder.fit_transform(pandas_df['USER_ID'])
    pandas_df['CONTENT_ID'] = i_encoder.fit_transform(pandas_df['CONTENT_ID'])

    # Drop duplicates
    df_unique_interactions = pandas_df.drop_duplicates(subset=['USER_ID', 'CONTENT_ID'])

    # Create a sparse matrix from the unique interactions
    sparse_matrix = coo_matrix(
        (df_unique_interactions['INTERACTION_COUNT'],
         (df_unique_interactions['USER_ID'], df_unique_interactions['CONTENT_ID']))
    ).tocsr()

    # print(sparse_matrix)
    return sparse_matrix


def create_model(session, brand, section):
    # First load some data
    print('getting dataset')

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    my_data = get_user_items(session, brand, section, user_encoder, item_encoder)
    print(my_data)
    print(my_data.shape)

    if brand == 'DRUMEO':
        num_factors = 15
    else:
        num_factors = 35

    # here we use Implicit's ALS algorithm
    rec_mod = AlternatingLeastSquares(factors=num_factors)

    # create an instance of the ImplicitRecommender class, which
    # serves as a wrapper for the Implicit ALS model.
    my_recsys = ImplicitRecommender(rec_mod)

    my_recsys.train(my_data)
    print('model trained')

    # session.sql("USE DATABASE ML_SCRATCH").collect()
    # session.sql("USE SCHEMA GMURR_MODELS").collect()

    table_name = "RECSYS_V2.CF_MODELS.%s_%s_MODELS" % (brand, section)

    create_table_sql = """
    CREATE OR REPLACE TABLE {} (
        model_id INT AUTOINCREMENT,
        model_data VARBINARY,
        user_encoder_data VARBINARY,
        item_encoder_data VARBINARY
    )
    """.format(table_name)

    session.sql(create_table_sql).collect()

    # Serialize the model to a byte stream
    model_byte_stream = io.BytesIO()
    joblib.dump(my_recsys, model_byte_stream)
    model_byte_stream.seek(0)

    # Convert to base64 for storing in VARBINARY
    model_base64 = base64.b64encode(model_byte_stream.getvalue())
    print('size of model: ', sys.getsizeof(model_base64))

    # Save user encoder
    ue_byte_stream = io.BytesIO()
    joblib.dump(user_encoder, ue_byte_stream)
    ue_byte_stream.seek(0)
    user_encoder_base64 = base64.b64encode(ue_byte_stream.getvalue())
    print('size of user encoder: ', sys.getsizeof(user_encoder_base64))

    # Save item encoder
    ie_byte_stream = io.BytesIO()
    joblib.dump(item_encoder, ie_byte_stream)
    ie_byte_stream.seek(0)
    item_encoder_base64 = base64.b64encode(ie_byte_stream.getvalue())
    print('size of item encoder: ', sys.getsizeof(item_encoder_base64))

    # Assume model_base64, user_encoder_base64, and item_encoder_base64 are already created
    print(table_name)
    # Insert the serialized model and encoders into Snowflake
    # Prepare your query with placeholders for parameters
    insert_query = """
        INSERT INTO {} (model_data, user_encoder_data, item_encoder_data) 
        VALUES (?, ?, ?)
    """.format(table_name)

    # try:
    # Execute the query with a tuple of parameters
    session.sql(insert_query, (model_base64, user_encoder_base64, item_encoder_base64)).collect()
    # except Exception as inst:
    #    print(type(inst))
    #    print(inst)

    return "Success"


def main(session: snowpark.Session):
    brands = ['DRUMEO', 'SINGEO', 'GUITAREO', 'PIANOTE']
    sections = ['COURSE', 'SONG', 'QUICK_TIPS', 'WORKOUT']

    for brand in brands:
        for section in sections:
            print('creating model for: ', brand, section)
            create_model(session, brand, section)

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

    brands = ['DRUMEO', 'SINGEO', 'GUITAREO', 'PIANOTE']
    sections = ['COURSE', 'SONG', 'QUICK_TIPS', 'WORKOUT']

    for brand in brands:
        for section in sections:
            create_model(session, brand, section)
