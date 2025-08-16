import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.visualization.visualize import desereliazer


def serializer(df, pickle_file_path):
    base_path = r"C:\Users\user\azercell_project1\src\data"
    pickle_file_path = os.path.join(base_path, pickle_file_path)
    try:
        with open(pickle_file_path, "wb") as f:
            pickle.dump(df, f)
        print(f"✅ DataFrame serialized to {pickle_file_path}")
    except Exception as e:
        print("❌ Error serializing DataFrame:", e)


def upload_to_processed_data(file):
    base_path = r"C:\Users\user\azercell_project1\src\data"
    pickle_file_path = os.path.join(base_path, file)
    target_path = r"C:\Users\user\azercell_project1\data\processed"
    target_path = os.path.join(target_path, file)
    shutil.copy(pickle_file_path, target_path)


def housing_prices_future():
    df = desereliazer("housing_prices.pkl")
    df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    df.columns = df.columns.str.lower()
    df.fillna({"price": df["price"].median()}, inplace=True)
    df["area"] = np.log1p(df["area"])
    df["room"] = np.log1p(df["room"])
    df["city"] = df.address.str.split(",").str[1]
    df["street"] = df.address.str.split(",").str[0].str.split(r"\d").str[0]
    df["address"] = df.address.str.split(",").str[0]
    df.drop(columns="address", inplace=True)
    df["district"] = df.zip.str.split(" ").str[0]
    df["sector"] = df.zip.str.split(" ").str[1]
    df.drop(columns="zip", inplace=True)
    print(df.head())
    serializer(df, "housing_prices.pkl")
    upload_to_processed_data("housing_prices.pkl")


def data_usage_production_future():
    df = desereliazer("data_usage_production.pkl")
    df.set_index("telephone_number", inplace=True)
    serializer(df, "data_usage_production.pkl")
    upload_to_processed_data("data_usage_production.pkl")


def sentence_transformer(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Fill missing variety texts (if any)
    df["variety"] = df["variety"].fillna("missing")
    # Encode all sentences at once (returns a 2D numpy array)
    embeddings = model.encode(df["variety"].tolist())
    # Convert embeddings array into DataFrame with multiple columns
    embedding_df = pd.DataFrame(
        embeddings,
        index=[i for i in range(1, len(embeddings) + 1)],
        columns=[f"emb_{i}" for i in range(1, embeddings.shape[1] + 1)],
    )
    return embedding_df


def ramen_ratings_future():
    df = desereliazer("ramen-ratings.pkl")
    df.set_index("Review #", inplace=True)
    df.drop("Top Ten", axis=1, inplace=True)
    df.columns = df.columns.str.lower()
    df["stars"] = df["stars"].replace("Unrated", np.nan)
    df["stars"] = df["stars"].fillna(df["stars"].mode()[0])
    embedding_df = sentence_transformer(df)
    X = embedding_df.values
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X_std)
    df_reduced = pd.DataFrame(X_pca, index=[i for i in range(1, 2580 + 1)])
    df = pd.concat([df.drop(columns=["variety"]), df_reduced], axis=1)
    df.columns = df.columns.astype(str)
    print(df.head())
    serializer(df, "ramen-ratings.pkl")
    upload_to_processed_data("ramen-ratings.pkl")


def multisim_dataset_future():
    df = desereliazer("multisim_dataset.pkl")
    df.set_index("telephone_number", inplace=True)
    serializer(df, "multisim_dataset.pkl")
    upload_to_processed_data("multisim_dataset.pkl")


def main():
    housing_prices_future()
    data_usage_production_future()
    ramen_ratings_future()
    multisim_dataset_future()


if __name__ == "__main__":
    main()
