import os
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np

from coinglass_api.api import CoinglassAPIv3, CoinglassAPIError, CoinglassRequestError, RateLimitExceededError


  

# Remove the load_config function and replace it with this:
def get_api_key():
    return os.environ['COINGLASS_API_KEY']

def plot_ohlc(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['c'], label='Close')
    plt.fill_between(df.index, df['l'], df['h'], alpha=0.3)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

def plot_funding_rates(df, title, filename):
    plt.figure(figsize=(15, 8))
    df_plot = df.copy()
    df_plot['funding_rate'] = pd.to_numeric(df_plot['funding_rate'], errors='coerce')
    df_plot = df_plot.dropna(subset=['funding_rate'])
    df_plot['abs_funding_rate'] = df_plot['funding_rate'].abs()
    df_plot = df_plot.sort_values('abs_funding_rate', ascending=False)

    if not df_plot.empty:
        exchanges = df_plot['exchange']
        funding_rates = df_plot['funding_rate']
        margin_types = df_plot['margin_type']

        x = np.arange(len(exchanges))
        width = 0.35

        fig, ax = plt.subplots(figsize=(15, 8))
        rects1 = ax.bar(x[margin_types == 'USDT/USD'] - width/2, funding_rates[margin_types == 'USDT/USD'], width, label='USDT/USD', color='b', alpha=0.7)
        rects2 = ax.bar(x[margin_types == 'Token'] + width/2, funding_rates[margin_types == 'Token'], width, label='Token', color='r', alpha=0.7)

        ax.set_ylabel('Funding Rate')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(exchanges, rotation=90)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"images/{filename}.png")
    else:
        print(f"No valid data to plot for {title}")
    plt.close()
    
def main():
    api_key = get_api_key()
    cg_api = CoinglassAPIv3(api_key)

    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    try:
        # # Supported coins
        # supported_coins_df = cg_api.supported_coins()
        # print("Supported coins:")
        # print(supported_coins_df)
        # plt.figure(figsize=(12, 6))
        # plt.bar(range(len(supported_coins_df)), [1]*len(supported_coins_df))
        # plt.title("Supported Coins")
        # plt.xlabel("Coins")
        # plt.ylabel("Count")
        # plt.xticks([])  # Remove x-axis labels as there are too many
        # plt.tight_layout()
        # plt.savefig("images/supported_coins.png")
        # plt.close()

        # # Supported exchanges and pairs
        # supported_pairs_df = cg_api.supported_exchange_pairs()
        # print("\nSupported exchanges and pairs:")
        # print(supported_pairs_df)
        # exchange_counts = supported_pairs_df['exchange'].value_counts()
        # plt.figure(figsize=(12, 6))
        # plt.bar(exchange_counts.index, exchange_counts.values)
        # plt.title("Supported Exchanges")
        # plt.xlabel("Exchange")
        # plt.ylabel("Number of Pairs")
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.savefig("images/supported_exchanges.png")
        # plt.close()

        # # OHLC history
        # ohlc_df = cg_api.ohlc_history(exchange="Binance", symbol="BTCUSDT", interval="1d", limit=30)
        # print("\nOHLC history:")
        # print(ohlc_df)
        # plt.figure(figsize=(12, 6))
        # plt.plot(ohlc_df.index, ohlc_df['c'])
        # plt.title("BTCUSDT OHLC History (Close Price)")
        # plt.xlabel("Date")
        # plt.ylabel("Close Price")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig("images/ohlc_history.png")
        # plt.close()

        # # Aggregated OHLC history
        # ohlc_agg_df = cg_api.ohlc_aggregated_history(symbol="BTC", interval="1d", limit=30)
        # print("\nAggregated OHLC history:")
        # print(ohlc_agg_df)
        # plt.figure(figsize=(12, 6))
        # plt.plot(ohlc_agg_df.index, ohlc_agg_df['c'])
        # plt.title("BTC Aggregated OHLC History (Close Price)")
        # plt.xlabel("Date")
        # plt.ylabel("Close Price")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig("images/ohlc_aggregated_history.png")
        # plt.close()

        # # Aggregated stablecoin margin OHLC history
        # ohlc_agg_stablecoin_df = cg_api.ohlc_aggregated_stablecoin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=30)
        # print("\nAggregated stablecoin margin OHLC history:")
        # print(ohlc_agg_stablecoin_df)
        # plt.figure(figsize=(12, 6))
        # plt.plot(ohlc_agg_stablecoin_df.index, ohlc_agg_stablecoin_df['c'])
        # plt.title("BTC Aggregated Stablecoin Margin OHLC History (Close Price)")
        # plt.xlabel("Date")
        # plt.ylabel("Close Price")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig("images/ohlc_agg_stablecoin_history.png")
        # plt.close()

        # # Aggregated coin margin OHLC history
        # ohlc_agg_coin_df = cg_api.ohlc_aggregated_coin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=30)
        # print("\nAggregated coin margin OHLC history:")
        # print(ohlc_agg_coin_df)
        # plt.figure(figsize=(12, 6))
        # plt.plot(ohlc_agg_coin_df.index, ohlc_agg_coin_df['c'])
        # plt.title("BTC Aggregated Coin Margin OHLC History (Close Price)")
        # plt.xlabel("Date")
        # plt.ylabel("Close Price")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig("images/ohlc_agg_coin_history.png")
        # plt.close()

        # # Open interest data from exchanges
        # exchange_list_df = cg_api.exchange_list(symbol="BTC")
        # print("\nOpen interest data from exchanges:")
        # print(exchange_list_df)
        # plt.figure(figsize=(12, 6))
        # plt.bar(exchange_list_df['exchange'], exchange_list_df['openInterest'])
        # plt.title("BTC Open Interest by Exchange")
        # plt.xlabel("Exchange")
        # plt.ylabel("Open Interest")
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.savefig("images/open_interest_by_exchange.png")
        # plt.close()

        # # Exchange history chart
        # exchange_history_df = cg_api.exchange_history_chart(symbol="BTC", range="4h", unit="USD")
        # print("\nExchange history chart:")
        # print(exchange_history_df)
        # plt.figure(figsize=(12, 6))
        # for column in exchange_history_df.columns:
        #     if column != "price":
        #         plt.plot(exchange_history_df.index, exchange_history_df[column], label=column)
        # plt.title("BTC Open Interest History by Exchange")
        # plt.xlabel("Date")
        # plt.ylabel("Open Interest")
        # plt.legend()
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig("images/exchange_history_chart.png")
        # plt.close()
        
    # # Funding Rate OHLC History
    #     funding_rate_ohlc_df = cg_api.funding_rate_ohlc_history(
    #         exchange="Binance", 
    #         symbol="BTCUSDT", 
    #         interval="1d", 
    #         limit=30
    #     )
    #     print("\nFunding Rate OHLC History:")
    #     print(funding_rate_ohlc_df)

    #     # Plot Funding Rate OHLC History
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(funding_rate_ohlc_df.index, funding_rate_ohlc_df['c'], label='Close')
    #     plt.fill_between(funding_rate_ohlc_df.index, funding_rate_ohlc_df['l'], funding_rate_ohlc_df['h'], alpha=0.3)
    #     plt.title("BTCUSDT Funding Rate OHLC History")
    #     plt.xlabel("Date")
    #     plt.ylabel("Funding Rate")
    #     plt.legend()
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig("images/funding_rate_ohlc_history.png")
    #     plt.close()
    
    # # OI Weight OHLC History
    #     oi_weight_ohlc_df = cg_api.oi_weight_ohlc_history(
    #         symbol="BTC", 
    #         interval="1d", 
    #         limit=30
    #     )
    #     print("\nOI Weight OHLC History:")
    #     print(oi_weight_ohlc_df)

    #     # Plot OI Weight OHLC History
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(oi_weight_ohlc_df.index, oi_weight_ohlc_df['c'], label='Close')
    #     plt.fill_between(oi_weight_ohlc_df.index, oi_weight_ohlc_df['l'], oi_weight_ohlc_df['h'], alpha=0.3)
    #     plt.title("BTC OI Weight OHLC History")
    #     plt.xlabel("Date")
    #     plt.ylabel("OI Weight")
    #     plt.legend()
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig("images/oi_weight_ohlc_history.png")
    #     plt.close()
    # # Vol Weight OHLC History
    #     vol_weight_ohlc_df = cg_api.vol_weight_ohlc_history(
    #         symbol="BTC", 
    #         interval="1d", 
    #         limit=30
    #     )
    #     print("\nVol Weight OHLC History:")
    #     print(vol_weight_ohlc_df)
    #     plot_ohlc(vol_weight_ohlc_df, "BTC Vol Weight OHLC History", "vol_weight_ohlc_history")
    
    # Funding Rate Exchange List
        funding_rate_df = cg_api.funding_rate_exchange_list(symbol="BTC")
        
        # Filter out rows with NaN funding rates
        funding_rate_df = funding_rate_df.dropna(subset=['funding_rate'])
        
        # Convert funding_rate to numeric and calculate absolute value
        funding_rate_df['funding_rate'] = pd.to_numeric(funding_rate_df['funding_rate'], errors='coerce')
        funding_rate_df['abs_funding_rate'] = funding_rate_df['funding_rate'].abs()
        
        print("\nFunding Rate Exchange List (Top 10 by absolute funding rate):")
        top_10 = funding_rate_df.sort_values('abs_funding_rate', ascending=False).head(10)
        print(top_10[['symbol', 'margin_type', 'exchange', 'funding_rate', 'next_funding_time']].to_string(index=False))

        # Plot Funding Rates (top 20 exchanges by absolute funding rate)
        top_20 = funding_rate_df.sort_values('abs_funding_rate', ascending=False).head(20)
        plot_funding_rates(top_20, "BTC Funding Rates - Top 20 Exchanges", "funding_rates_top_20")

        # Display next funding times
        print("\nNext Funding Times (next 10):")
        next_funding_times = funding_rate_df[["exchange", "next_funding_time"]].sort_values("next_funding_time")
        next_funding_times = next_funding_times[next_funding_times["next_funding_time"].notna()]  # Remove rows with NaT
        print(next_funding_times.head(10).to_string(index=False))

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