import time
import pandas as pd
import tensorflow as tf
from libreco.algorithms import ALS, NGCF, LightGCN
from libreco.data import DatasetPure, split_by_ratio_chrono


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


def get_model_recommendations_df(model, model_name, n_rec=10):
    """
    For each user, call recommend_user(...) to get top-n_rec items.
    Build and return a DataFrame with columns:
        user_id | movie_id | recommendation_rank | module_source
    """
    all_users = trainset_df["user"].unique().tolist()
    rec_dict = model.recommend_user(user=all_users, n_rec=n_rec)

    df_rows = []
    for user_id, items in rec_dict.items():
        for rank, item_id in enumerate(items, start=1):
            df_rows.append((user_id, item_id, rank, model_name))

    recommendations_df = pd.DataFrame(
        df_rows,
        columns=["user_id", "movie_id", "recommendation_rank", "module_source"]
    )

    return recommendations_df


if __name__ == "__main__":

    Path = "../Dataset/100k/train_test_sets"

    # Load the ratings data
    trainset_df = pd.read_csv(f'{Path}/u1.base',
                              sep='\t',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'],
                              encoding='latin-1')

    testset_df = pd.read_csv(f'{Path}/u1.test',
                             sep='\t',
                             names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             encoding='latin-1')

    # ------------------------------------------------
    # ------------------------------------------------
    # 1) Rename columns
    trainset_df.rename(
        columns={
            "user_id": "user",
            "movie_id": "item",
            "rating": "label"
        },
        inplace=True
    )

    testset_df.rename(
        columns={
            "user_id": "user",
            "movie_id": "item",
            "rating": "label"
        },
        inplace=True
    )

    # 2) Re-order columns if needed (e.g., drop the timestamp or put it last)
    trainset_df = trainset_df[["user", "item", "label", "timestamp"]]
    testset_df = testset_df[["user", "item", "label", "timestamp"]]

    # Suppose after renaming columns to 'user' and 'item'
    train_users = set(trainset_df["user"].unique())
    train_items = set(trainset_df["item"].unique())

    # Filter test so it only has user/item that appear in train
    testset_df = testset_df[
        testset_df["user"].isin(train_users) & testset_df["item"].isin(train_items)
        ]

    # ------------------------------------------------
    # ------------------------------------------------

    # Build train and eval sets
    train_data, data_info = DatasetPure.build_trainset(trainset_df)
    eval_data = DatasetPure.build_evalset(testset_df)
    print(data_info)

    metrics = [
        "loss",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "map",
        "ndcg",
    ]

    # ------------------------------------------------------------
    # NGCF
    # ------------------------------------------------------------
    reset_state("NGCF")
    ngcf = NGCF(
        task="ranking",
        data_info=data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=3e-4,
        lr_decay=False,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        node_dropout=0.1,
        message_dropout=0.1,
        hidden_units=(64, 64, 64),
        device="cuda",
    )
    ngcf.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("NGCF prediction (user=1, item=2333): ", ngcf.predict(user=1, item=2333))
    print("NGCF recommendation (user=1): ", ngcf.recommend_user(user=1, n_rec=7))

    # ------------------------------------------------------------
    # LightGCN
    # ------------------------------------------------------------
    reset_state("LightGCN")
    lightgcn = LightGCN(
        task="ranking",
        data_info=data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=20,
        lr=3e-4,
        lr_decay=False,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.1,
        n_layers=3,
        device="cuda",
    )
    lightgcn.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("LightGCN prediction (user=1, item=2333): ", lightgcn.predict(user=1, item=2333))
    print("LightGCN recommendation (user=1): ", lightgcn.recommend_user(user=1, n_rec=7))

    # ------------------------------------------------------------
    # Build a unified recommendations DataFrame
    # ------------------------------------------------------------
    # 1) Gather all user_ids from the training set
    all_train_users = trainset_df["user"].unique()

    # 2) NGCF recommendations
    ngcf_recs_df = get_model_recommendations_df(
        model=ngcf,
        model_name="NGCF",
        n_rec=10  # or whatever number of recommendations you prefer
    )

    # 3) LightGCN recommendations
    lightgcn_recs_df = get_model_recommendations_df(
        model=lightgcn,
        model_name="LightGCN",
        n_rec=10
    )



    # 4) Concatenate both
    recommendations_df = pd.concat([ngcf_recs_df, lightgcn_recs_df], ignore_index=True)

    # 5) Convert columns to the desired data types
    recommendations_df = recommendations_df.astype({
        'user_id': 'int',
        'movie_id': 'int',
        'recommendation_rank': 'int',
        'module_source': 'str'
    })

    print("\nFinal Recommendations DataFrame:\n", recommendations_df.head(20))
