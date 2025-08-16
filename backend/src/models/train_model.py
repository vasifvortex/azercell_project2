import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile, f_regression, chi2
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import pickle
from category_encoders import CatBoostEncoder
import optuna
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.visualization.visualize import desereliazer
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


def train_model(df):
    y = df["data_compl_usg_local_m1"]
    X = df.drop("data_compl_usg_local_m1", axis=1)

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

    # Select only columns you want
    X = X[numeric_features + categorical_features]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Pipelines for numeric and categorical
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("encoder", CatBoostEncoder()),
            ("selector", SelectPercentile(f_regression, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=13, random_state=42)),
        ]
    )

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Model test score: {score:.3f}")

    return clf


def save_model(model, model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def data_usage_production_train():
    df = desereliazer("data_usage_production.pkl")
    model = train_model(df)
    base_path = r"C:\Users\user\azercell_project1\src\models"
    model_path = os.path.join(base_path, "data_usage_production_train_model.pkl")
    save_model(model, model_path)


def ramen_ratings_train_model(df):
    y = df["stars"]
    X = df.drop("stars", axis=1)

    numeric_features = X.columns[X.dtypes == "float32"]
    categorical_features = ["brand", "style", "country"]

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    # Select only columns you want
    X = X[numeric_features + categorical_features]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Pipelines for numeric and categorical
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", CatBoostEncoder()),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=400),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    print("model score: %.3f" % clf.score(X_test, y_test))

    return clf


def ramen_ratings_train():
    df = desereliazer("ramen-ratings.pkl")
    model = ramen_ratings_train_model(df)
    base_path = r"C:\Users\user\azercell_project1\src\models"
    model_path = os.path.join(base_path, "ramen-ratings_train_model.pkl")
    save_model(model, model_path)


def multisim_dataset_train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X, y = shuffle(X, y, random_state=42)
    numeric_features = X.columns[X.dtypes == "float64"].append(X.columns[X.dtypes == "int64"])
    categorical_features = X.columns[X.dtypes == "object"]

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    # Select only columns you want
    X = X[numeric_features + categorical_features]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Pipelines for numeric and categorical
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("yeo-johnson_transformer", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", CatBoostEncoder()),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    penalty="l1", C=0.5, solver="liblinear", max_iter=1000, random_state=42
                ),
            ),
        ]
    )
    Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=5))]
    )
    Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )
    Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )

    def objective(trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 1e-2),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 1e-2),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        clf6 = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", XGBClassifier(**param))]
        )
        scores = cross_val_score(clf6, X, y, cv=3, scoring="accuracy")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print("Best Parameters:", study.best_params)
    # After tuning, train final model with best params
    best_params = study.best_params
    best_params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})

    final_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", XGBClassifier(**best_params))]
    )

    final_pipeline.fit(X_train, y_train)
    print("Final model test accuracy:", final_pipeline.score(X_test, y_test))

    return final_pipeline


def multisim_dataset_train():
    df = desereliazer("multisim_dataset.pkl")
    model = multisim_dataset_train_model(df)
    base_path = r"C:\Users\user\azercell_project1\src\models"
    model_path = os.path.join(base_path, "multisim_dataset_train_model.pkl")
    save_model(model, model_path)


def main():
    data_usage_production_train()
    ramen_ratings_train()
    multisim_dataset_train()


if __name__ == "__main__":
    main()
