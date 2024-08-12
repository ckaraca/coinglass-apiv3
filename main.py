import os
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

from coinglass_api.api import CoinglassAPI, CoinglassAPIError, CoinglassRequestError, RateLimitExceededError


  

"""
def get_coinglass_data(url, api_key, symbol="BTC"):
    headers = {
        "accept": "application/json",
        "coinglassSecret": api_key
    }
    params = {"symbol": symbol}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()['data']

def create_plots(cem):
    plt.figure(figsize=(16, 14))
    sns.set_style("whitegrid")

    # 1. Bar plot of Open Interest by Exchange
    plt.subplot(2, 2, 1)
    cem_sorted = cem.sort_values('openInterest', ascending=False)
    sns.barplot(x='openInterest', y='exchangeName', data=cem_sorted)
    plt.title('Open Interest by Exchange')
    plt.xlabel('Open Interest')
    plt.ylabel('Exchange')

    # 2. Scatter plot of Open Interest vs Volume
    plt.subplot(2, 2, 2)
    plt.scatter(cem['openInterest'], cem['volUsd'])
    plt.title('Open Interest vs Volume')
    plt.xlabel('Open Interest')
    plt.ylabel('Volume (USD)')
    for i, txt in enumerate(cem['exchangeName']):
        plt.annotate(txt, (cem['openInterest'].iloc[i], cem['volUsd'].iloc[i]))

    # 3. Heatmap of Correlation between numerical columns
    plt.subplot(2, 2, 3)
    numeric_cols = cem.select_dtypes(include=[np.number]).columns
    correlation = cem[numeric_cols].corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numerical Columns')

    # 4. Bar plot of 24h Change by Exchange
    plt.subplot(2, 2, 4)
    cem_sorted_change = cem.sort_values('h24Change', ascending=True)
    sns.barplot(x='h24Change', y='exchangeName', data=cem_sorted_change)
    plt.title('24h Change by Exchange')
    plt.xlabel('24h Change (%)')
    plt.ylabel('Exchange')

    plt.tight_layout()
    plt.savefig('coinglass_data_plot.png', dpi=300)
    print("Plot has been saved as 'coinglass_data_plot.png'")

def main():
    try:
        config = load_config()
        data = get_coinglass_data(config['coinglass_url'], config['coinglass_apikey'])
        
        cem = pd.DataFrame(data)
        print(cem.head())
        print(cem.shape)

        create_plots(cem)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Exception details:")
        print(sys.exc_info())

    finally:
        plt.close()  # Close the plot to free up memory

"""
# Remove the load_config function and replace it with this:
def get_api_key():
    return os.environ['COINGLASS_API_KEY']

def main():
    api_key = get_api_key()
    cg_api = CoinglassAPI(api_key)
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
        
        # Fetch and print the aggregated stablecoin margin OHLC history
        ohlc_agg_stablecoin_df = cg_api.ohlc_aggregated_stablecoin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=10)
        print(ohlc_agg_stablecoin_df)
        
        # Fetch and print the aggregated coin margin OHLC history
        ohlc_agg_coin_df = cg_api.ohlc_aggregated_coin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=10)
        print(ohlc_agg_coin_df)
        
        # Fetch and print the open interest data from exchanges
        exchange_list_df = cg_api.exchange_list(symbol="BTC")
        print(exchange_list_df)
        
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
    
