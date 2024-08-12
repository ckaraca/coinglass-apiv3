import os
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
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
    
def plot_liquidation_history(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['longLiquidationUsd'], label='Long Liquidations', color='green')
    plt.plot(df.index, df['shortLiquidationUsd'], label='Short Liquidations', color='red')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Liquidation Value (USD)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()
    
def plot_top_liquidations(df, title, filename, n=10):
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('liquidationUsd24h', ascending=False).head(n)
    
    plt.bar(df_sorted['symbol'], df_sorted['longLiquidationUsd24h'], label='Long Liquidations', color='green')
    plt.bar(df_sorted['symbol'], df_sorted['shortLiquidationUsd24h'], bottom=df_sorted['longLiquidationUsd24h'], label='Short Liquidations', color='red')
    
    plt.title(title)
    plt.xlabel("Coin")
    plt.ylabel("Liquidation Value (USD)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

def plot_exchange_liquidations(df, title, filename):
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('liquidationUsd', ascending=True)
    df_sorted = df_sorted[df_sorted['exchange'] != 'All']  # Exclude 'All' from the plot
    
    plt.barh(df_sorted['exchange'], df_sorted['longLiquidationUsd'], label='Long Liquidations', color='green')
    plt.barh(df_sorted['exchange'], df_sorted['shortLiquidationUsd'], left=df_sorted['longLiquidationUsd'], label='Short Liquidations', color='red')
    
    plt.title(title)
    plt.xlabel("Liquidation Value (USD)")
    plt.ylabel("Exchange")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()
    
def plot_liquidation_heatmap(data, title, filename):
    y = data['y']
    liq = data['liq']
    prices = data['prices']

    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(y), len(prices)))
    for x, y_idx, value in liq:
        heatmap_data[y_idx][x] = value

    # Create a DataFrame for the price data
    price_df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='s')
    price_df = price_df.set_index('timestamp')

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    # Custom colormap
    colors = ['#3a0ca3', '#4361ee', '#4cc9f0', '#7bf1a8', '#ccff33']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot the heatmap
    im = ax1.imshow(heatmap_data, aspect='auto', cmap=cmap, norm=LogNorm(vmin=1e4), extent=[0, len(prices)-1, y[0], y[-1]], origin='lower')
    
    # Plot the price
    ax1_twin = ax1.twinx()
    ax1_twin.plot(np.arange(len(prices)), price_df['close'], color='white', linewidth=1.5)

    # Set labels and title
    ax1.set_ylabel('Price')
    ax1_twin.set_ylabel('Price')
    ax1.set_title(title)

    # Set x-axis ticks (dates)
    num_x_ticks = 6
    x_tick_locations = np.linspace(0, len(prices)-1, num_x_ticks).astype(int)
    ax2.set_xticks(x_tick_locations)
    ax2.set_xticklabels(price_df.index[x_tick_locations].strftime('%m-%d %H:%M'), rotation=45, ha='right')

    # Set y-axis ticks (prices)
    num_y_ticks = 8
    y_tick_locations = np.linspace(y[0], y[-1], num_y_ticks)
    ax1.set_yticks(y_tick_locations)
    ax1.set_yticklabels([f"${int(p)}" for p in y_tick_locations])
    ax1_twin.set_yticks(y_tick_locations)
    ax1_twin.set_yticklabels([f"${int(p)}" for p in y_tick_locations])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Liquidation Value')

    # Plot volume in the bottom subplot
    ax2.bar(np.arange(len(prices)), price_df['volume'], color='#4361ee', alpha=0.7)
    ax2.set_ylabel('Volume')

    # Improve overall clarity
    ax1.grid(False)
    ax1_twin.grid(False)
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png", dpi=300, bbox_inches='tight')
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
    
    # # Funding Rate Exchange List
    #     funding_rate_df = cg_api.funding_rate_exchange_list(symbol="BTC")
        
    #     # Filter out rows with NaN funding rates
    #     funding_rate_df = funding_rate_df.dropna(subset=['funding_rate'])
        
    #     # Convert funding_rate to numeric and calculate absolute value
    #     funding_rate_df['funding_rate'] = pd.to_numeric(funding_rate_df['funding_rate'], errors='coerce')
    #     funding_rate_df['abs_funding_rate'] = funding_rate_df['funding_rate'].abs()
        
    #     print("\nFunding Rate Exchange List (Top 10 by absolute funding rate):")
    #     top_10 = funding_rate_df.sort_values('abs_funding_rate', ascending=False).head(10)
    #     print(top_10[['symbol', 'margin_type', 'exchange', 'funding_rate', 'next_funding_time']].to_string(index=False))

    #     # Plot Funding Rates (top 20 exchanges by absolute funding rate)
    #     top_20 = funding_rate_df.sort_values('abs_funding_rate', ascending=False).head(20)
    #     plot_funding_rates(top_20, "BTC Funding Rates - Top 20 Exchanges", "funding_rates_top_20")

    #     # Display next funding times
    #     print("\nNext Funding Times (next 10):")
    #     next_funding_times = funding_rate_df[["exchange", "next_funding_time"]].sort_values("next_funding_time")
    #     next_funding_times = next_funding_times[next_funding_times["next_funding_time"].notna()]  # Remove rows with NaT
    #     print(next_funding_times.head(10).to_string(index=False))
        
    # # Liquidation History
    #     liq_history_df = cg_api.liquidation_history(
    #         exchange="Binance",
    #         symbol="BTCUSDT",
    #         interval="1d",
    #         limit=30
    #     )
    #     print("\nLiquidation History:")
    #     print(liq_history_df)

    #     # Plot Liquidation History
    #     plot_liquidation_history(liq_history_df, "BTCUSDT Liquidation History", "liquidation_history")

    #     # Calculate and print total liquidations
    #     total_long_liq = liq_history_df['longLiquidationUsd'].sum()
    #     total_short_liq = liq_history_df['shortLiquidationUsd'].sum()
    #     print(f"\nTotal Long Liquidations: ${total_long_liq:,.2f}")
    #     print(f"Total Short Liquidations: ${total_short_liq:,.2f}")
    #     print(f"Total Liquidations: ${total_long_liq + total_short_liq:,.2f}")

    #     # Find and print the day with the highest liquidations
    #     liq_history_df['totalLiquidationUsd'] = liq_history_df['longLiquidationUsd'] + liq_history_df['shortLiquidationUsd']
    #     max_liq_day = liq_history_df.loc[liq_history_df['totalLiquidationUsd'].idxmax()]
    #     print(f"\nDay with highest liquidations: {max_liq_day.name.date()}")
    #     print(f"Total liquidations on that day: ${max_liq_day['totalLiquidationUsd']:,.2f}")
    #     print(f"Long liquidations: ${max_liq_day['longLiquidationUsd']:,.2f}")
    #     print(f"Short liquidations: ${max_liq_day['shortLiquidationUsd']:,.2f}")
        
    # # Aggregated Liquidation History
    #     agg_liq_history_df = cg_api.liquidation_aggregated_history(
    #         symbol="BTC",
    #         interval="1d",
    #         limit=30
    #     )
    #     print("\nAggregated Liquidation History (BTC):")
    #     print(agg_liq_history_df)

    #     # Plot Aggregated Liquidation History
    #     plot_liquidation_history(agg_liq_history_df, "BTC Aggregated Liquidation History", "liquidation_history_aggregated")

    #     # Calculate and print total liquidations for both datasets
    #     for name, df in [("Binance BTCUSDT", liq_history_df), ("Aggregated BTC", agg_liq_history_df)]:
    #         total_long_liq = df['longLiquidationUsd'].sum()
    #         total_short_liq = df['shortLiquidationUsd'].sum()
    #         print(f"\nTotal Liquidations for {name}:")
    #         print(f"Total Long Liquidations: ${total_long_liq:,.2f}")
    #         print(f"Total Short Liquidations: ${total_short_liq:,.2f}")
    #         print(f"Total Liquidations: ${total_long_liq + total_short_liq:,.2f}")

    #         # Find and print the day with the highest liquidations
    #         df['totalLiquidationUsd'] = df['longLiquidationUsd'] + df['shortLiquidationUsd']
    #         max_liq_day = df.loc[df['totalLiquidationUsd'].idxmax()]
    #         print(f"\nDay with highest liquidations for {name}: {max_liq_day.name.date()}")
    #         print(f"Total liquidations on that day: ${max_liq_day['totalLiquidationUsd']:,.2f}")
    #         print(f"Long liquidations: ${max_liq_day['longLiquidationUsd']:,.2f}")
    #         print(f"Short liquidations: ${max_liq_day['shortLiquidationUsd']:,.2f}")

    # # Liquidation Coin List
    #     liq_coin_list_df = cg_api.liquidation_coin_list(ex="Binance")
    #     print("\nLiquidation Coin List (Binance):")
    #     print(liq_coin_list_df)

    #     # Plot Top 10 Coins by 24h Liquidation
    #     plot_top_liquidations(liq_coin_list_df, "Top 10 Coins by 24h Liquidation (Binance)", "top_10_liquidations_24h")

    #     # Calculate and print total liquidations
    #     total_liq_24h = liq_coin_list_df['liquidationUsd24h'].sum()
    #     total_long_liq_24h = liq_coin_list_df['longLiquidationUsd24h'].sum()
    #     total_short_liq_24h = liq_coin_list_df['shortLiquidationUsd24h'].sum()

    #     print(f"\nTotal Liquidations (24h) on Binance:")
    #     print(f"Total Liquidations: ${total_liq_24h:,.2f}")
    #     print(f"Total Long Liquidations: ${total_long_liq_24h:,.2f}")
    #     print(f"Total Short Liquidations: ${total_short_liq_24h:,.2f}")

    #     # Find and print the coin with the highest liquidations
    #     max_liq_coin = liq_coin_list_df.loc[liq_coin_list_df['liquidationUsd24h'].idxmax()]
    #     print(f"\nCoin with highest liquidations (24h): {max_liq_coin['symbol']}")
    #     print(f"Total liquidations: ${max_liq_coin['liquidationUsd24h']:,.2f}")
    #     print(f"Long liquidations: ${max_liq_coin['longLiquidationUsd24h']:,.2f}")
    #     print(f"Short liquidations: ${max_liq_coin['shortLiquidationUsd24h']:,.2f}")

    #     # Calculate and print the ratio of long to short liquidations
    #     liq_coin_list_df['long_short_ratio'] = liq_coin_list_df['longLiquidationUsd24h'] / liq_coin_list_df['shortLiquidationUsd24h']
    #     print("\nTop 5 coins with highest long/short liquidation ratio:")
    #     print(liq_coin_list_df.sort_values('long_short_ratio', ascending=False)[['symbol', 'long_short_ratio']].head())

    #     print("\nTop 5 coins with lowest long/short liquidation ratio:")
    #     print(liq_coin_list_df.sort_values('long_short_ratio', ascending=True)[['symbol', 'long_short_ratio']].head())

    # # Liquidation Exchange List
    #     liq_exchange_list_df = cg_api.liquidation_exchange_list(symbol="BTC", range="24h")
    #     print("\nLiquidation Exchange List (BTC, 24h):")
    #     print(liq_exchange_list_df)

    #     # Plot Exchange Liquidations
    #     plot_exchange_liquidations(liq_exchange_list_df, "BTC Liquidations by Exchange (24h)", "btc_liquidations_by_exchange")

    #     # Calculate and print total liquidations
    #     total_row = liq_exchange_list_df[liq_exchange_list_df['exchange'] == 'All']
    #     if not total_row.empty:
    #         total_row = total_row.iloc[0]
    #         print(f"\nTotal BTC Liquidations (24h):")
    #         print(f"Total Liquidations: ${total_row['liquidationUsd']:,.2f}")
    #         print(f"Total Long Liquidations: ${total_row['longLiquidationUsd']:,.2f}")
    #         print(f"Total Short Liquidations: ${total_row['shortLiquidationUsd']:,.2f}")
    #     else:
    #         print("\nNo 'All' row found in the data.")

    #     # Find and print the exchange with the highest liquidations
    #     exchange_data = liq_exchange_list_df[liq_exchange_list_df['exchange'] != 'All'].copy()
    #     if not exchange_data.empty:
    #         max_liq_exchange = exchange_data.loc[exchange_data['liquidationUsd'].idxmax()]
    #         print(f"\nExchange with highest BTC liquidations (24h): {max_liq_exchange['exchange']}")
    #         print(f"Total liquidations: ${max_liq_exchange['liquidationUsd']:,.2f}")
    #         print(f"Long liquidations: ${max_liq_exchange['longLiquidationUsd']:,.2f}")
    #         print(f"Short liquidations: ${max_liq_exchange['shortLiquidationUsd']:,.2f}")
    #     else:
    #         print("\nNo exchange data available.")

    #     # Calculate and print the ratio of long to short liquidations for each exchange
    #     exchange_data.loc[:, 'long_short_ratio'] = exchange_data['longLiquidationUsd'] / exchange_data['shortLiquidationUsd']
    #     print("\nLong/Short liquidation ratio by exchange:")
    #     ratio_data = exchange_data[['exchange', 'long_short_ratio']].sort_values('long_short_ratio', ascending=False)
    #     print(ratio_data)

    #     # Additional analysis: Percentage of liquidations by exchange
    #     if not exchange_data.empty:
    #         total_liq = exchange_data['liquidationUsd'].sum()
    #         exchange_data.loc[:, 'liquidation_percentage'] = (exchange_data['liquidationUsd'] / total_liq) * 100
    #         print("\nPercentage of liquidations by exchange:")
    #         print(exchange_data[['exchange', 'liquidation_percentage']].sort_values('liquidation_percentage', ascending=False))
    
    # Liquidation Aggregated Heatmap Model2
        liq_heatmap_data = cg_api.liquidation_aggregated_heatmap_model2(symbol="BTC", range="3d")
        print("\nLiquidation Aggregated Heatmap Data (BTC, 3d):")
        print(json.dumps(liq_heatmap_data, indent=2))

        # Plot Liquidation Heatmap
        plot_liquidation_heatmap(liq_heatmap_data, "BTC Liquidation Heatmap (3d)", "btc_liquidation_heatmap")
        

        # Additional analysis
        total_liquidation = sum(value for _, _, value in liq_heatmap_data['liq'])
        print(f"\nTotal potential liquidation value: ${total_liquidation:,.2f}")

        # Find the price level with the highest liquidation value
        max_liq = max(liq_heatmap_data['liq'], key=lambda x: x[2])
        max_liq_price = liq_heatmap_data['y'][max_liq[1]]
        print(f"Price level with highest liquidation value: ${max_liq_price:,.2f}")
        print(f"Highest liquidation value: ${max_liq[2]:,.2f}")

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