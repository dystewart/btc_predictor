import pandas as pd
import os

def load_features(filepath):
    """
    Load the processed feature dataset with indicators.
    """
    return pd.read_csv(filepath, index_col="timestamp", parse_dates=True)

def add_price_direction_labels(df):
    """
    Add binary classification labels based on future close price.
    - 1 if future price is greater than now
    - 0 otherwise
    """
    # Shifted close prices for future targets
    df["close_t+1m"] = df["close"].shift(-1)
    df["close_t+5m"] = df["close"].shift(-5)

    # Binary labels for direction
    df["target_1m"] = (df["close_t+1m"] > df["close"]).astype(int)
    df["target_5m"] = (df["close_t+5m"] > df["close"]).astype(int)

    # Drop the helper columns and final NaNs caused by shifting
    df.drop(columns=["close_t+1m", "close_t+5m"], inplace=True)
    df.dropna(inplace=True)

    return df

def save_labeled_data(df, output_path):
    """
    Save the labeled DataFrame to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved labeled data to {output_path}")

if __name__ == "__main__":
    input_path = "data/processed/btc_1m_features.csv"
    output_path = "data/processed/btc_1m_labeled.csv"

    df = load_features(input_path)
    df = add_price_direction_labels(df)
    save_labeled_data(df, output_path)
