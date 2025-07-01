# Import required libraries
import ccxt           # Crypto exchange library to access market data
import pandas as pd   # For data manipulation and analysis
import os

# Define a function to fetch 1-minute OHLCV (Open, High, Low, Close, Volume) data
def fetch_btc_ohlcv_1m(limit=1000):
    # Create an instance of the Kraken exchange using ccxt
    exchange = ccxt.kraken()
    
    try:
        # Fetch the latest 1-minute candlestick data for BTC/USDT
        # `limit=1000` fetches up to 1000 minutes of data (~16.5 hours)
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=1000)

        # Convert the raw data into a pandas DataFrame with named columns
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert the timestamp from UNIX milliseconds to readable datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set the timestamp column as the DataFrame index
        df.set_index('timestamp', inplace=True)

        # Ensure all numeric columns are of float type for precision and consistency
        df = df.astype(float)

        # Return the final cleaned and formatted DataFrame
        return df

    # If there's an error (e.g., network issue, API error), catch and print it
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None  # Return None if fetching fails

# Example usage of the function:
# df = fetch_btc_ohlcv_1m()  # Fetch the data and store it in `df`

# Print the last few rows to verify output
# print(df.tail())
def save_to_csv(df, filepath):
    # Create parent directories if not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save DataFrame to CSV
    df.to_csv(filepath)
    print(f"Saved data to {filepath}")

if __name__ == "__main__":
    df = fetch_btc_ohlcv_1m()
    if df is not None:
        save_to_csv(df, "data/raw/btc_1m.csv")
