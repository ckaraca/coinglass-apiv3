import os
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

from coinglass_api.api import CoinglassAPIv3, CoinglassAPIError, CoinglassRequestError, RateLimitExceededError


  

# Remove the load_config function and replace it with this:
def get_api_key():
    return os.environ['COINGLASS_API_KEY']

def main():
    api_key = get_api_key()
    cg_api = CoinglassAPIv3(api_key)

    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    try:
        # Supported coins
        supported_coins_df = cg_api.supported_coins()
        print("Supported coins:")
        print(supported_coins_df)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(supported_coins_df)), [1]*len(supported_coins_df))
        plt.title("Supported Coins")
        plt.xlabel("Coins")
        plt.ylabel("Count")
        plt.xticks([])  # Remove x-axis labels as there are too many
        plt.tight_layout()
        plt.savefig("images/supported_coins.png")
        plt.close()

        # Supported exchanges and pairs
        supported_pairs_df = cg_api.supported_exchange_pairs()
        print("\nSupported exchanges and pairs:")
        print(supported_pairs_df)
        exchange_counts = supported_pairs_df['exchange'].value_counts()
        plt.figure(figsize=(12, 6))
        plt.bar(exchange_counts.index, exchange_counts.values)
        plt.title("Supported Exchanges")
        plt.xlabel("Exchange")
        plt.ylabel("Number of Pairs")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("images/supported_exchanges.png")
        plt.close()

        # OHLC history
        ohlc_df = cg_api.ohlc_history(exchange="Binance", symbol="BTCUSDT", interval="1d", limit=30)
        print("\nOHLC history:")
        print(ohlc_df)
        plt.figure(figsize=(12, 6))
        plt.plot(ohlc_df.index, ohlc_df['c'])
        plt.title("BTCUSDT OHLC History (Close Price)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/ohlc_history.png")
        plt.close()

        # Aggregated OHLC history
        ohlc_agg_df = cg_api.ohlc_aggregated_history(symbol="BTC", interval="1d", limit=30)
        print("\nAggregated OHLC history:")
        print(ohlc_agg_df)
        plt.figure(figsize=(12, 6))
        plt.plot(ohlc_agg_df.index, ohlc_agg_df['c'])
        plt.title("BTC Aggregated OHLC History (Close Price)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/ohlc_aggregated_history.png")
        plt.close()

        # Aggregated stablecoin margin OHLC history
        ohlc_agg_stablecoin_df = cg_api.ohlc_aggregated_stablecoin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=30)
        print("\nAggregated stablecoin margin OHLC history:")
        print(ohlc_agg_stablecoin_df)
        plt.figure(figsize=(12, 6))
        plt.plot(ohlc_agg_stablecoin_df.index, ohlc_agg_stablecoin_df['c'])
        plt.title("BTC Aggregated Stablecoin Margin OHLC History (Close Price)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/ohlc_agg_stablecoin_history.png")
        plt.close()

        # Aggregated coin margin OHLC history
        ohlc_agg_coin_df = cg_api.ohlc_aggregated_coin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=30)
        print("\nAggregated coin margin OHLC history:")
        print(ohlc_agg_coin_df)
        plt.figure(figsize=(12, 6))
        plt.plot(ohlc_agg_coin_df.index, ohlc_agg_coin_df['c'])
        plt.title("BTC Aggregated Coin Margin OHLC History (Close Price)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/ohlc_agg_coin_history.png")
        plt.close()

        # Open interest data from exchanges
        exchange_list_df = cg_api.exchange_list(symbol="BTC")
        print("\nOpen interest data from exchanges:")
        print(exchange_list_df)
        plt.figure(figsize=(12, 6))
        plt.bar(exchange_list_df['exchange'], exchange_list_df['openInterest'])
        plt.title("BTC Open Interest by Exchange")
        plt.xlabel("Exchange")
        plt.ylabel("Open Interest")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("images/open_interest_by_exchange.png")
        plt.close()

        # Exchange history chart
        exchange_history_df = cg_api.exchange_history_chart(symbol="BTC", range="4h", unit="USD")
        print("\nExchange history chart:")
        print(exchange_history_df)
        plt.figure(figsize=(12, 6))
        for column in exchange_history_df.columns:
            if column != "price":
                plt.plot(exchange_history_df.index, exchange_history_df[column], label=column)
        plt.title("BTC Open Interest History by Exchange")
        plt.xlabel("Date")
        plt.ylabel("Open Interest")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/exchange_history_chart.png")
        plt.close()

    except CoinglassRequestError as e:
        print(f"Request error: {e}")
    except RateLimitExceededError as e:
        print("Rate limit exceeded. Please try again later.")
    except CoinglassAPIError as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()    
