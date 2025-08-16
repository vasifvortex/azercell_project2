import os
import pickle
from sklearn.utils import shuffle
from fastapi import HTTPException


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_data(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        df = pickle.load(f)
    return df


def test_model(model, df):
    X = df.drop("data_compl_usg_local_m1", axis=1)

    # Select features same as training
    numeric_features = [
        "refill_total_m2",
        "refill_total_m3",
        "frequency",
        "recency",
        "tenure",
        "lastrefillamount_m2",
        "tot_inact_status_days_l1m_m2",
        "refill_total_m4",
        "data_compl_usg_local_m2",
        "data_compl_usg_local_m3",
        "data_compl_usg_local_m4",
    ]
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    X = X[numeric_features + categorical_features]
    y_pred = model.predict(X)
    return y_pred.tolist()


def ramen_ratings_test_model(model, df):
    X = df.drop("stars", axis=1)

    numeric_features = X.columns[X.dtypes == "float32"]
    categorical_features = ["brand", "style", "country"]

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    # Select only columns you want
    X = X[numeric_features + categorical_features]
    y_pred = model.predict(X)
    return y_pred.tolist()


def multisim_dataset_test_model(model, df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X, y = shuffle(X, y, random_state=42)
    numeric_features = X.columns[X.dtypes == "float64"].append(
        X.columns[X.dtypes == "int64"]
    )
    categorical_features = X.columns[X.dtypes == "object"]

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    # Select only columns you want
    X = X[numeric_features + categorical_features]
    y_pred = model.predict(X)
    return y_pred.tolist()


def data_usage_production_test():
    base_path = "src/data"
    model_path = "src/models"
    model_path = os.path.join(model_path, "data_usage_production_train_model.pkl")
    data_path = os.path.join(base_path, "data_usage_production.pkl")

    model = load_model(model_path)
    df = load_data(data_path)
    return list(test_model(model, df))


def ramen_ratings_test():
    base_path = "src/data"
    model_path = "src/models"
    model_path = os.path.join(model_path, "ramen-ratings_train_model.pkl")
    data_path = os.path.join(base_path, "ramen-ratings.pkl")

    model = load_model(model_path)
    df = load_data(data_path)
    return list(ramen_ratings_test_model(model, df))


def multisim_dataset_test():
    base_path = "src/data"
    model_path = "src/models"
    model_path = os.path.join(model_path, "multisim_dataset_train_model.pkl")
    data_path = os.path.join(base_path, "multisim_dataset.pkl")

    model = load_model(model_path)
    df = load_data(data_path)
    return list(multisim_dataset_test_model(model, df))


def main(file_content, filename: str):
    if filename == "data_usage_production.parquet":
        predictions = data_usage_production_test()
    elif filename == "ramen-ratings.csv":
        predictions = ramen_ratings_test()
    elif filename == "multisim_dataset.parquet":
        predictions = multisim_dataset_test()
    else:
        raise HTTPException(
            status_code=422,
            detail="Use datasets: multisim_dataset.parquet, ramen-ratings.csv, data_usage_production.parquet",
        )
    return predictions


if __name__ == "__main__":
    main()
