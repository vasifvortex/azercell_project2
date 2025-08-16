def data_downloader(local_file_path, pickle_file_path):
    import os
    import pickle
    import pandas as pd

    base_path = r"C:\Users\user\azercell_project1\src\data"
    local_file_path = os.path.join(base_path, local_file_path)
    pickle_file_path = os.path.join(base_path, pickle_file_path)

    # Step 1: Load into a DataFrame
    try:
        df = pd.read_csv(local_file_path)
        print("✅ CSV loaded into DataFrame")

    except Exception:
        try:
            df = pd.read_parquet(local_file_path)
            print("✅ Parquet loaded into DataFrame")
        except Exception as e:
            print("❌ Error loading CSV:", e)
            exit()

    # Step 2: Serialize DataFrame with pickle
    try:
        with open(pickle_file_path, "wb") as f:
            pickle.dump(df, f)
        print(f"✅ DataFrame serialized to {pickle_file_path}")
    except Exception as e:
        print("❌ Error serializing DataFrame:", e)


def main():
    data_downloader("HousingPrices-Amsterdam-August-2021.csv", "housing_prices.pkl")
    data_downloader("data_usage_production.parquet", "data_usage_production.pkl")
    data_downloader("ramen-ratings.csv", "ramen-ratings.pkl")
    data_downloader("multisim_dataset.parquet", "multisim_dataset.pkl")


if __name__ == "__main__":
    main()
