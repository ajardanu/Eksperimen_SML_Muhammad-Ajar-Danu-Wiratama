import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Telco Customer Churn dataset
    """
    df = df.copy()

    # Drop ID column
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges datatype
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features and target
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline
    """
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def preprocess_data(
    input_path: str,
    output_dir: str = "namadataset_preprocessing"
):
    """
    Full preprocessing pipeline:
    - load data
    - clean data
    - build & fit preprocessor
    - transform data
    - save processed dataset & preprocessor
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load
    df = load_data(input_path)

    # Clean
    df = clean_data(df)

    # Split
    X, y = split_features_target(df)

    # Preprocessing
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    # Save processed data
    processed_df = pd.DataFrame(
        X_processed.toarray()
        if hasattr(X_processed, "toarray")
        else X_processed
    )

    processed_df["Churn"] = y.values
    processed_path = os.path.join(output_dir, "telco_preprocessed.csv")
    processed_df.to_csv(processed_path, index=False)

    # Save preprocessor
    joblib.dump(
        preprocessor,
        os.path.join(output_dir, "preprocessor.pkl")
    )

    print("Preprocessing selesai!")
    print(f"Dataset tersimpan di: {processed_path}")
    print("Preprocessor tersimpan sebagai: preprocessor.pkl")


if __name__ == "__main__":
    INPUT_PATH = "namadataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_DIR = "namadataset_preprocessing"

    preprocess_data(INPUT_PATH, OUTPUT_DIR)