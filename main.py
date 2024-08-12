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
    try:
        # # Fetch and print the supported coins
        # supported_coins_df = cg_api.supported_coins()
        # print(supported_coins_df)
        
        # # Fetch and print the supported exchanges and pairs
        # supported_pairs_df = cg_api.supported_exchange_pairs()
        # print(supported_pairs_df)
        
        # # Fetch and print the OHLC history
        # ohlc_df = cg_api.ohlc_history(exchange="Binance", symbol="BTCUSDT", interval="1d", limit=10)
        # print(ohlc_df)
        
        # # Fetch and print the aggregated OHLC history
        # ohlc_agg_df = cg_api.ohlc_aggregated_history(symbol="BTC", interval="1d", limit=10)
        # print(ohlc_agg_df)
        
        # # Fetch and print the aggregated stablecoin margin OHLC history
        # ohlc_agg_stablecoin_df = cg_api.ohlc_aggregated_stablecoin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=10)
        # print(ohlc_agg_stablecoin_df)
        
        # # Fetch and print the aggregated coin margin OHLC history
        # ohlc_agg_coin_df = cg_api.ohlc_aggregated_coin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=10)
        # print(ohlc_agg_coin_df)
        
        # # Fetch and print the open interest data from exchanges
        # exchange_list_df = cg_api.exchange_list(symbol="BTC")
        # print(exchange_list_df)
        
        # Fetch and print the exchange history chart
        exchange_history_df = cg_api.exchange_history_chart(symbol="BTC", range="4h", unit="USD")
        print(exchange_history_df)
        
        
        
    except CoinglassRequestError as e:
        print(f"Request error: {e}")
    except RateLimitExceededError as e:
        print("Rate limit exceeded. Please try again later.")
    except CoinglassAPIError as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
    
