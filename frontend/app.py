import time
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# -------------------- Streamlit Page --------------------
st.set_page_config(
    page_title="DataMinds'25 - ML Predictor & EDA",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖",
)


# -------------------- Utility Functions --------------------
def detect_mime(filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        return "text/csv"
    if name.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if name.endswith(".xls"):
        return "application/vnd.ms-excel"
    if name.endswith(".parquet"):
        return "text/parquet"
    return "application/octet-stream"


def load_df_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame | None:
    """Load CSV, Excel, or Parquet bytes into a DataFrame."""
    try:
        bio = BytesIO(file_bytes)
        if filename.lower().endswith(".csv"):
            return pd.read_csv(bio)
        elif filename.lower().endswith(".parquet"):
            return pd.read_parquet(bio)
        elif filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(bio)
        else:
            st.error(
                "Unsupported file format. Please upload a CSV, Excel, or Parquet file."
            )
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def send_to_api(file_bytes: bytes, filename: str, api_url: str) -> dict | None:
    """Send the raw uploaded file to FastAPI /predict endpoint."""
    try:
        files = {"file": (filename, file_bytes, detect_mime(filename))}
        resp = requests.post(api_url, files=files, timeout=60)
        if resp.headers.get("content-type", "").startswith("application/json"):
            data = resp.json()
        else:
            st.error(
                f"Unexpected response from API (status {resp.status_code}): {resp.text[:400]}"
            )
            return None

        if resp.status_code == 200:
            return data
        else:
            detail = data.get("detail") if isinstance(data, dict) else None
            st.error(f"API Error {resp.status_code}: {detail or data}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while calling API: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# -------------------- EDA Functions --------------------
def make_plots(column, df):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.subplot(1, 2, 2)
    sns.histplot(df[column], bins=10)
    plt.title(column + " Distribution")
    plt.tight_layout()
    st.pyplot(plt)


def perform_eda(df):
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(100), use_container_width=True, height=300)

    if st.checkbox("📈 Show Data Statistics"):
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        make_plots(col, df)

    # Heatmap for numeric correlation
    if len(numeric_cols) >= 2:
        st.markdown("### 🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True)
        st.pyplot(plt)


# -------------------- Main App --------------------
def main():
    # --- Session State Initialization ---
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = None
    if "file_bytes" not in st.session_state:
        st.session_state.file_bytes = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "results" not in st.session_state:
        st.session_state.results = None

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            "<h2>🤖 DataMinds'25 ML Predictor & EDA</h2>", unsafe_allow_html=True
        )
        st.markdown("---")
        api_url = "http://backend:8000/predict"
        uploaded = st.file_uploader(
            "Choose a CSV, Excel, or Parquet file",
            type=["csv", "xlsx", "xls", "parquet"],
        )

        if uploaded is not None and uploaded != st.session_state.uploaded:
            st.session_state.uploaded = uploaded
            st.session_state.file_bytes = uploaded.getvalue()
            st.session_state.df = load_df_from_bytes(
                st.session_state.file_bytes, uploaded.name
            )
            st.session_state.results = None
        elif uploaded is None:
            st.session_state.uploaded = None
            st.session_state.file_bytes = None
            st.session_state.df = None
            st.session_state.results = None

    # --- Main Content ---
    st.title("🤖 DataMinds'25 ML Predictor & EDA")

    if st.session_state.df is not None:
        st.success("File uploaded successfully!")

        # Show EDA
        st.markdown("---")
        st.header("🔍 Exploratory Data Analysis (EDA)")
        perform_eda(st.session_state.df)

        # Predict Button
        st.markdown("---")
        st.header("🔮 Generate Predictions")
        if st.button("🚀 Predict"):
            with st.spinner("Generating predictions..."):
                start = time.time()
                data = send_to_api(
                    st.session_state.file_bytes,
                    st.session_state.uploaded.name,
                    api_url,
                )
                if data and isinstance(data, dict) and data.get("status") == "success":
                    preds = data.get("data", {}).get("predictions", [])
                    n = min(len(preds), len(st.session_state.df))
                    st.session_state.results = pd.DataFrame(st.session_state.df.head(n))
                    st.session_state.results["Prediction"] = preds[:n]
                    st.success(
                        f"✅ Predictions generated in {round(time.time() - start, 2)}s"
                    )
                else:
                    st.error("Prediction failed.")

        # Show predictions
        if st.session_state.results is not None:
            st.markdown("### 📋 Prediction Results")
            st.dataframe(st.session_state.results, use_container_width=True)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=st.session_state.results.to_csv(index=False),
                file_name=f"predictions_{int(time.time())}.csv",
                mime="text/csv",
            )
    else:
        st.info("📁 Upload a file to start EDA and generate predictions.")


if __name__ == "__main__":
    main()
