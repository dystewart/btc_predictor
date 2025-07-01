import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_labeled_data(path):
    """Load labeled data from CSV."""
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)

def train_xgboost_classifier(df, target_col="target_1m"):
    """
    Train XGBoost classifier to predict price direction.
    """
    # Drop rows with NaNs (safety check)
    df = df.dropna()

    # Split features and target
    X = df.drop(columns=["target_1m", "target_5m", "close"])  # drop labels and raw price
    y = df[target_col]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # no shuffle = time-consistent split
    )

    # Train model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

if __name__ == "__main__":
    filepath = "data/processed/btc_1m_labeled.csv"
    df = load_labeled_data(filepath)

    model = train_xgboost_classifier(df)
