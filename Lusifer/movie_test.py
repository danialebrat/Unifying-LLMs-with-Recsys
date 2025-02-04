import pandas as pd



movie_path = f"../Dataset/100k/movies_enriched_dataset.pkl"
movies_df = pd.read_pickle(movie_path)

print(movies_df["movie_info"].values)