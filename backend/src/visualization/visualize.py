import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def desereliazer(pickle_file_path):
    import os
    import pickle

    base_path = "src/data"
    pickle_file_path = os.path.join(base_path, pickle_file_path)

    with open(pickle_file_path, "rb") as f:
        loaded_df = pickle.load(f)
    return loaded_df


def make_plots(column, df):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.subplot(1, 2, 2)
    sns.histplot(df[column], bins=10)
    plt.title(column + " Distribution")
    plt.tight_layout()
    plt.show()


def housing_prices_eda():
    df = desereliazer("housing_prices.pkl")
    print(df.head())
    make_plots("price", df)
    make_plots("area", df)
    make_plots("room", df)
    sns.scatterplot(x="area", y="price", data=df)
    sns.scatterplot(x="room", y="price", data=df)
    sns.heatmap(df[["price", "area", "room"]].corr(), annot=True)
    plt.show()


def multisim_dataset_eda():
    df = desereliazer("multisim_dataset.pkl")
    print(df.head())
    sns.countplot(x="trf", data=df)
    sns.countplot(x="gndr", data=df)
    sns.countplot(x="is_dualsim", data=df)
    sns.countplot(x="target", data=df)
    pd.crosstab(df["gndr"], df["target"], normalize="index")
    pd.crosstab(df["trf"], df["target"], normalize="index")
    pd.crosstab(df["is_dualsim"], df["target"], normalize="index")
    sns.heatmap(
        df[
            [
                "target",
                "tenure",
                "val4_1",
                "val6_1",
                "val8_1",
                "val10_1",
                "val19_1",
                "val20_1",
                "val21_1",
            ]
        ].corr(),
        annot=True,
    )
    plt.show()


def main():
    housing_prices_eda()
    multisim_dataset_eda()


if __name__ == "__main__":
    main()
