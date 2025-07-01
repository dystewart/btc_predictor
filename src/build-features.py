import pandas as pd
import os

# Import specific indicator classes from the 'ta' package
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

def load_raw_data(filepath):
    """
    Load raw OHLCV data from CSV file.
    """
    return pd.read_csv(filepath, index_col="timestamp", parse_dates=True)

def add_technical_indicators(df):
    """
    Add technical indicators using the 'ta' library.
    """
    # RSI (14-period)
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi_14"] = rsi.rsi()

    # MACD (12/26) and signal line
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # EMA 20 and EMA 50
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()

    # Rolling standard deviation (volatility proxy)
    df["volatility"] = df["close"].rolling(window=10).std()

    # Drop rows with NaNs (from indicator warm-up)
    df.dropna(inplace=True)
    return df

def save_features(df, output_path):
    """
    Save the processed DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    input_path = "data/raw/btc_1m.csv"
    output_path = "data/processed/btc_1m_features.csv"

    df = load_raw_data(input_path)
    df = add_technical_indicators(df)
    save_features(df, output_path)