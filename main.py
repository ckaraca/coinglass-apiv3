import os

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import importlib
import coinglass_api.api
importlib.reload(coinglass_api.api)
from coinglass_api.api import CoinglassAPIv3, CoinglassAPIError, CoinglassRequestError, RateLimitExceededError


from plot_utils import (
    plot_ohlc,
    plot_funding_rates,
    plot_liquidation_history,
    plot_top_liquidations,
    plot_exchange_liquidations,
    plot_liquidation_heatmap,
    plot_long_short_ratio,
    plot_liquidation_heatmap_v2,
    plot_aggregated_taker_buy_sell_ratio,
    plot_taker_buy_sell_volume_ratio,
    plot_aggregated_taker_buy_sell_volume,
    plot_exchange_taker_buy_sell_ratio,
    plot_coins_markets,
    plot_pairs_markets,
    plot_orderbook_history,
    plot_aggregated_orderbook_history,
    plot_spot_taker_buy_sell_volume,
    plot_spot_aggregated_taker_buy_sell_volume,
    plot_spot_orderbook_history,
    plot_spot_orderbook_history,
    plot_spot_pairs_markets,
    plot_bitcoin_bubble_index,
    plot_ahr999_index,
    plot_two_year_ma_multiplier,
    plot_200_week_ma_heatmap,
    plot_puell_multiple,
    plot_stock_to_flow,
    plot_pi_cycle_top_indicator,
    plot_golden_ratio_multiplier,
    plot_bitcoin_profitable_days,
    plot_bitcoin_rainbow_chart,
    plot_fear_greed_index   
)
  

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
        
        # Supported exchanges and pairs
        supported_pairs_df = cg_api.supported_exchange_pairs()
        print("\nSupported exchanges and pairs:")
        print(supported_pairs_df)
        

        # OHLC history
        ohlc_df = cg_api.ohlc_history(exchange="Binance", symbol="BTCUSDT", interval="1h", limit=3500)
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
        
        # plot total liquidation value accoss exchanges
        

        # Aggregated OHLC history
        ohlc_agg_df = cg_api.ohlc_aggregated_history(symbol="BTC", interval="1h", limit=30)
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
        ohlc_agg_stablecoin_df = cg_api.ohlc_aggregated_stablecoin_margin_history(exchanges="Binance", symbol="BTC", interval="1h", limit=30)
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
        ohlc_agg_coin_df = cg_api.ohlc_aggregated_coin_margin_history(exchanges="Binance", symbol="BTC", interval="1h", limit=30)
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
        
    # Funding Rate OHLC History
        funding_rate_ohlc_df = cg_api.funding_rate_ohlc_history(
            exchange="Binance", 
            symbol="BTCUSDT", 
            interval="1d", 
            limit=30
        )
        print("\nFunding Rate OHLC History:")
        print(funding_rate_ohlc_df)

        # Plot Funding Rate OHLC History
        plt.figure(figsize=(12, 6))
        plt.plot(funding_rate_ohlc_df.index, funding_rate_ohlc_df['c'], label='Close')
        plt.fill_between(funding_rate_ohlc_df.index, funding_rate_ohlc_df['l'], funding_rate_ohlc_df['h'], alpha=0.3)
        plt.title("BTCUSDT Funding Rate OHLC History")
        plt.xlabel("Date")
        plt.ylabel("Funding Rate")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/funding_rate_ohlc_history.png")
        plt.close()
    
    # OI Weight OHLC History
        oi_weight_ohlc_df = cg_api.oi_weight_ohlc_history(
            symbol="BTC", 
            interval="1d", 
            limit=30
        )
        print("\nOI Weight OHLC History:")
        print(oi_weight_ohlc_df)

        # Plot OI Weight OHLC History
        plt.figure(figsize=(12, 6))
        plt.plot(oi_weight_ohlc_df.index, oi_weight_ohlc_df['c'], label='Close')
        plt.fill_between(oi_weight_ohlc_df.index, oi_weight_ohlc_df['l'], oi_weight_ohlc_df['h'], alpha=0.3)
        plt.title("BTC OI Weight OHLC History")
        plt.xlabel("Date")
        plt.ylabel("OI Weight")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/oi_weight_ohlc_history.png")
        plt.close()
    # Vol Weight OHLC History
        vol_weight_ohlc_df = cg_api.vol_weight_ohlc_history(
            symbol="BTC", 
            interval="1d", 
            limit=30
        )
        print("\nVol Weight OHLC History:")
        print(vol_weight_ohlc_df)
        plot_ohlc(vol_weight_ohlc_df, "BTC Vol Weight OHLC History", "vol_weight_ohlc_history")
    
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
        
    # Liquidation History
        liq_history_df = cg_api.liquidation_history(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=30
        )
        print("\nLiquidation History:")
        print(liq_history_df)

        # Plot Liquidation History
        plot_liquidation_history(liq_history_df, "BTCUSDT Liquidation History", "liquidation_history")

        # Calculate and print total liquidations
        total_long_liq = liq_history_df['longLiquidationUsd'].sum()
        total_short_liq = liq_history_df['shortLiquidationUsd'].sum()
        print(f"\nTotal Long Liquidations: ${total_long_liq:,.2f}")
        print(f"Total Short Liquidations: ${total_short_liq:,.2f}")
        print(f"Total Liquidations: ${total_long_liq + total_short_liq:,.2f}")

        # Find and print the day with the highest liquidations
        liq_history_df['totalLiquidationUsd'] = liq_history_df['longLiquidationUsd'] + liq_history_df['shortLiquidationUsd']
        max_liq_day = liq_history_df.loc[liq_history_df['totalLiquidationUsd'].idxmax()]
        print(f"\nDay with highest liquidations: {max_liq_day.name.date()}")
        print(f"Total liquidations on that day: ${max_liq_day['totalLiquidationUsd']:,.2f}")
        print(f"Long liquidations: ${max_liq_day['longLiquidationUsd']:,.2f}")
        print(f"Short liquidations: ${max_liq_day['shortLiquidationUsd']:,.2f}")
        
    # Aggregated Liquidation History
        agg_liq_history_df = cg_api.liquidation_aggregated_history(
            symbol="BTC",
            interval="1h",
            limit=30
        )
        print("\nAggregated Liquidation History (BTC):")
        print(agg_liq_history_df)

        # Plot Aggregated Liquidation History
        plot_liquidation_history(agg_liq_history_df, "BTC Aggregated Liquidation History", "liquidation_history_aggregated")

        # Calculate and print total liquidations for both datasets
        for name, df in [("Binance BTCUSDT", liq_history_df), ("Aggregated BTC", agg_liq_history_df)]:
            total_long_liq = df['longLiquidationUsd'].sum()
            total_short_liq = df['shortLiquidationUsd'].sum()
            print(f"\nTotal Liquidations for {name}:")
            print(f"Total Long Liquidations: ${total_long_liq:,.2f}")
            print(f"Total Short Liquidations: ${total_short_liq:,.2f}")
            print(f"Total Liquidations: ${total_long_liq + total_short_liq:,.2f}")

            # Find and print the day with the highest liquidations
            df['totalLiquidationUsd'] = df['longLiquidationUsd'] + df['shortLiquidationUsd']
            max_liq_day = df.loc[df['totalLiquidationUsd'].idxmax()]
            print(f"\nDay with highest liquidations for {name}: {max_liq_day.name.date()}")
            print(f"Total liquidations on that day: ${max_liq_day['totalLiquidationUsd']:,.2f}")
            print(f"Long liquidations: ${max_liq_day['longLiquidationUsd']:,.2f}")
            print(f"Short liquidations: ${max_liq_day['shortLiquidationUsd']:,.2f}")

    # Liquidation Coin List
        liq_coin_list_df = cg_api.liquidation_coin_list(ex="Binance")
        print("\nLiquidation Coin List (Binance):")
        print(liq_coin_list_df)

        # Plot Top 10 Coins by 24h Liquidation
        plot_top_liquidations(liq_coin_list_df, "Top 10 Coins by 24h Liquidation (Binance)", "top_10_liquidations_24h")

        # Calculate and print total liquidations
        total_liq_24h = liq_coin_list_df['liquidationUsd24h'].sum()
        total_long_liq_24h = liq_coin_list_df['longLiquidationUsd24h'].sum()
        total_short_liq_24h = liq_coin_list_df['shortLiquidationUsd24h'].sum()

        print(f"\nTotal Liquidations (24h) on Binance:")
        print(f"Total Liquidations: ${total_liq_24h:,.2f}")
        print(f"Total Long Liquidations: ${total_long_liq_24h:,.2f}")
        print(f"Total Short Liquidations: ${total_short_liq_24h:,.2f}")

        # Find and print the coin with the highest liquidations
        max_liq_coin = liq_coin_list_df.loc[liq_coin_list_df['liquidationUsd24h'].idxmax()]
        print(f"\nCoin with highest liquidations (24h): {max_liq_coin['symbol']}")
        print(f"Total liquidations: ${max_liq_coin['liquidationUsd24h']:,.2f}")
        print(f"Long liquidations: ${max_liq_coin['longLiquidationUsd24h']:,.2f}")
        print(f"Short liquidations: ${max_liq_coin['shortLiquidationUsd24h']:,.2f}")

        # Calculate and print the ratio of long to short liquidations
        liq_coin_list_df['long_short_ratio'] = liq_coin_list_df['longLiquidationUsd24h'] / liq_coin_list_df['shortLiquidationUsd24h']
        print("\nTop 5 coins with highest long/short liquidation ratio:")
        print(liq_coin_list_df.sort_values('long_short_ratio', ascending=False)[['symbol', 'long_short_ratio']].head())

        print("\nTop 5 coins with lowest long/short liquidation ratio:")
        print(liq_coin_list_df.sort_values('long_short_ratio', ascending=True)[['symbol', 'long_short_ratio']].head())

    # Liquidation Exchange List
        liq_exchange_list_df = cg_api.liquidation_exchange_list(symbol="BTC", range="24h")
        print("\nLiquidation Exchange List (BTC, 24h):")
        print(liq_exchange_list_df)

        # Plot Exchange Liquidations
        plot_exchange_liquidations(liq_exchange_list_df, "BTC Liquidations by Exchange (24h)", "btc_liquidations_by_exchange")

        # Calculate and print total liquidations
        total_row = liq_exchange_list_df[liq_exchange_list_df['exchange'] == 'All']
        if not total_row.empty:
            total_row = total_row.iloc[0]
            print(f"\nTotal BTC Liquidations (24h):")
            print(f"Total Liquidations: ${total_row['liquidationUsd']:,.2f}")
            print(f"Total Long Liquidations: ${total_row['longLiquidationUsd']:,.2f}")
            print(f"Total Short Liquidations: ${total_row['shortLiquidationUsd']:,.2f}")
        else:
            print("\nNo 'All' row found in the data.")

        # Find and print the exchange with the highest liquidations
        exchange_data = liq_exchange_list_df[liq_exchange_list_df['exchange'] != 'All'].copy()
        if not exchange_data.empty:
            max_liq_exchange = exchange_data.loc[exchange_data['liquidationUsd'].idxmax()]
            print(f"\nExchange with highest BTC liquidations (24h): {max_liq_exchange['exchange']}")
            print(f"Total liquidations: ${max_liq_exchange['liquidationUsd']:,.2f}")
            print(f"Long liquidations: ${max_liq_exchange['longLiquidationUsd']:,.2f}")
            print(f"Short liquidations: ${max_liq_exchange['shortLiquidationUsd']:,.2f}")
        else:
            print("\nNo exchange data available.")

        # Calculate and print the ratio of long to short liquidations for each exchange
        exchange_data.loc[:, 'long_short_ratio'] = exchange_data['longLiquidationUsd'] / exchange_data['shortLiquidationUsd']
        print("\nLong/Short liquidation ratio by exchange:")
        ratio_data = exchange_data[['exchange', 'long_short_ratio']].sort_values('long_short_ratio', ascending=False)
        print(ratio_data)

        # Additional analysis: Percentage of liquidations by exchange
        if not exchange_data.empty:
            total_liq = exchange_data['liquidationUsd'].sum()
            exchange_data.loc[:, 'liquidation_percentage'] = (exchange_data['liquidationUsd'] / total_liq) * 100
            print("\nPercentage of liquidations by exchange:")
            print(exchange_data[['exchange', 'liquidation_percentage']].sort_values('liquidation_percentage', ascending=False))
    
    
    
         # Liquidation Heatmap Model2
        liq_heatmap_data = cg_api.liquidation_heatmap_model2(exchange="Binance", symbol="BTCUSDT", range="12h")
        print("\nLiquidation Heatmap Data (BTCUSDT, 12h):")
        for key, df in liq_heatmap_data.items():
            print(f"\n{key.capitalize()} DataFrame:")
            print(df.head())

        # Plot Liquidation Heatmap
        plot_liquidation_heatmap_v2(liq_heatmap_data, "BTCUSDT Liquidation Heatmap (12h)", "btc_liquidation_heatmap_v2", "12h")

        # # Additional analysis
        # total_liquidation = liq_heatmap_data['liquidations']['liquidation_value'].sum()
        # print(f"\nTotal potential liquidation value: ${total_liquidation:,.2f}")

        # # Find the price level with the highest liquidation value
        # max_liq = liq_heatmap_data['liquidations'].loc[liq_heatmap_data['liquidations']['liquidation_value'].idxmax()]
        # print(f"Price level with highest liquidation value: ${max_liq['price']:,.2f}")
        # print(f"Highest liquidation value: ${max_liq['liquidation_value']:,.2f}")

        # # Analyze liquidation distribution
        # price_ranges = pd.cut(liq_heatmap_data['liquidations']['price'], bins=5)
        # liq_distribution = liq_heatmap_data['liquidations'].groupby(price_ranges)['liquidation_value'].sum()
        # print("\nLiquidation distribution across price ranges:")
        # print(liq_distribution)

    
    # Global Long/Short Account Ratio
        global_long_short_ratio_df = cg_api.global_long_short_account_ratio(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nGlobal Long/Short Account Ratio (BTCUSDT, last 7 days):")
        print(global_long_short_ratio_df)

        # Top Accounts Long/Short Ratio
        top_long_short_ratio_df = cg_api.top_long_short_account_ratio(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nTop Accounts Long/Short Ratio (BTCUSDT, last 7 days):")
        print(top_long_short_ratio_df)

        # Plot Long/Short Account Ratio
        plot_long_short_ratio(global_long_short_ratio_df, top_long_short_ratio_df, 
                              "BTCUSDT Long/Short Account Ratio", "btc_long_short_ratio")

        # Additional analysis
        for df_name, df in [("Global", global_long_short_ratio_df), ("Top Accounts", top_long_short_ratio_df)]:
            current_ratio = df.iloc[-1]
            print(f"\nCurrent {df_name} Long/Short Account Ratio:")
            print(f"Long Account: {current_ratio['longAccount']:.2f}%")
            print(f"Short Account: {current_ratio['shortAccount']:.2f}%")
            print(f"Long/Short Ratio: {current_ratio['longShortRatio']:.3f}")

            # Calculate average ratio over the period
            avg_ratio = df['longShortRatio'].mean()
            print(f"\nAverage {df_name} Long/Short Ratio over the last 7 days: {avg_ratio:.3f}")

            # Find highest and lowest ratios
            max_ratio = df['longShortRatio'].max()
            min_ratio = df['longShortRatio'].min()
            print(f"Highest {df_name} Long/Short Ratio: {max_ratio:.3f}")
            print(f"Lowest {df_name} Long/Short Ratio: {min_ratio:.3f}")

        # Compare global and top account ratios
        correlation = global_long_short_ratio_df['longShortRatio'].corr(top_long_short_ratio_df['longShortRatio'])
        print(f"\nCorrelation between Global and Top Accounts Long/Short Ratios: {correlation:.3f}")
        
    # Aggregated Taker Buy/Sell Volume Ratio History
        agg_taker_ratio_df = cg_api.aggregated_taker_buy_sell_volume_ratio_history(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nAggregated Taker Buy/Sell Volume Ratio History (BTCUSDT, last 7 days):")
        print(agg_taker_ratio_df)

        # Plot Aggregated Taker Buy/Sell Volume Ratio
        plot_aggregated_taker_buy_sell_ratio(agg_taker_ratio_df, 
                                             "BTCUSDT Aggregated Taker Buy/Sell Volume Ratio", 
                                             "btc_aggregated_taker_ratio")

        # Additional analysis
        current_ratio = agg_taker_ratio_df.iloc[-1]['longShortRatio']
        print(f"\nCurrent Aggregated Taker Buy/Sell Volume Ratio: {current_ratio:.4f}")

        # Calculate average ratio over the period
        avg_ratio = agg_taker_ratio_df['longShortRatio'].mean()
        print(f"Average Aggregated Taker Buy/Sell Volume Ratio over the last 7 days: {avg_ratio:.4f}")

        # Find highest and lowest ratios
        max_ratio = agg_taker_ratio_df['longShortRatio'].max()
        min_ratio = agg_taker_ratio_df['longShortRatio'].min()
        print(f"Highest Aggregated Taker Buy/Sell Volume Ratio: {max_ratio:.4f}")
        print(f"Lowest Aggregated Taker Buy/Sell Volume Ratio: {min_ratio:.4f}")

        # Calculate percentage of time the ratio was above 1 (more buyers than sellers)
        buyers_dominant = (agg_taker_ratio_df['longShortRatio'] > 1).mean() * 100
        print(f"Percentage of time buyers were dominant: {buyers_dominant:.2f}%")
        
        # Aggregated Taker Buy/Sell Volume History
        agg_volume_df = cg_api.aggregated_taker_buy_sell_volume_history(
            exchanges="Binance,OKX,Bybit",
            symbol="BTC",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nAggregated Taker Buy/Sell Volume History (BTC, last 7 days):")
        print(agg_volume_df)

        # Plot Aggregated Taker Buy/Sell Volume
        plot_aggregated_taker_buy_sell_volume(agg_volume_df, 
                                            "BTC Aggregated Taker Buy/Sell Volume", 
                                            "btc_aggregated_taker_volume")

        # Additional analysis
        total_buy_volume = agg_volume_df['buy'].sum()
        total_sell_volume = agg_volume_df['sell'].sum()
        print(f"\nTotal Buy Volume: {total_buy_volume:,.2f}")
        print(f"Total Sell Volume: {total_sell_volume:,.2f}")

        buy_sell_ratio = total_buy_volume / total_sell_volume
        print(f"Buy/Sell Ratio: {buy_sell_ratio:.4f}")

        # Calculate percentage of time buy volume exceeds sell volume
        buy_dominant = (agg_volume_df['buy'] > agg_volume_df['sell']).mean() * 100
        print(f"Percentage of time buy volume exceeded sell volume: {buy_dominant:.2f}%")

        # Find the hour with the highest buy and sell volumes
        max_buy_hour = agg_volume_df.loc[agg_volume_df['buy'].idxmax()]
        max_sell_hour = agg_volume_df.loc[agg_volume_df['sell'].idxmax()]
        
        print(f"\nHour with highest buy volume: {max_buy_hour.name}")
        print(f"Buy volume: {max_buy_hour['buy']:,.2f}, Sell volume: {max_buy_hour['sell']:,.2f}")
    
        # Taker Buy/Sell Volume History
        taker_volume_df = cg_api.taker_buy_sell_volume_history(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nTaker Buy/Sell Volume History (BTCUSDT on Binance, last 7 days):")
        print(taker_volume_df)

        # Plot Taker Buy/Sell Volume and Ratio
        plot_taker_buy_sell_volume_ratio(taker_volume_df, 
                                        "BTCUSDT Taker Buy/Sell Volume on Binance", 
                                        "btcusdt_taker_volume")

        # Additional analysis
        total_buy_volume = taker_volume_df['buy'].sum()
        total_sell_volume = taker_volume_df['sell'].sum()
        print(f"\nTotal Buy Volume: {total_buy_volume:,.2f}")
        print(f"Total Sell Volume: {total_sell_volume:,.2f}")

        overall_buy_sell_ratio = total_buy_volume / total_sell_volume
        print(f"Overall Buy/Sell Ratio: {overall_buy_sell_ratio:.4f}")

        # Calculate percentage of time buy volume exceeds sell volume
        buy_dominant = (taker_volume_df['buy'] > taker_volume_df['sell']).mean() * 100
        print(f"Percentage of time buy volume exceeded sell volume: {buy_dominant:.2f}%")

        # Find the hour with the highest buy and sell volumes
        max_buy_hour = taker_volume_df.loc[taker_volume_df['buy'].idxmax()]
        max_sell_hour = taker_volume_df.loc[taker_volume_df['sell'].idxmax()]
        
        print(f"\nHour with highest buy volume: {max_buy_hour.name}")
        print(f"Buy volume: {max_buy_hour['buy']:,.2f}, Sell volume: {max_buy_hour['sell']:,.2f}")
        
        print(f"\nHour with highest sell volume: {max_sell_hour.name}")
        print(f"Buy volume: {max_sell_hour['buy']:,.2f}, Sell volume: {max_sell_hour['sell']:,.2f}")

        # Calculate and display average buy/sell ratio
        avg_ratio = (taker_volume_df['buy'] / taker_volume_df['sell']).mean()
        print(f"\nAverage Buy/Sell Ratio: {avg_ratio:.4f}")

        # Find and display the highest and lowest ratios
        taker_volume_df['ratio'] = taker_volume_df['buy'] / taker_volume_df['sell']
        max_ratio = taker_volume_df['ratio'].max()
        min_ratio = taker_volume_df['ratio'].min()
        print(f"Highest Buy/Sell Ratio: {max_ratio:.4f}")
        print(f"Lowest Buy/Sell Ratio: {min_ratio:.4f}")
        
        # Exchange Taker Buy/Sell Ratio
        exchange_taker_ratio = cg_api.exchange_taker_buy_sell_ratio(symbol="BTC", range="1h")
        print("\nExchange Taker Buy/Sell Ratio (BTC, last 1h):")
        print(f"Overall Buy Ratio: {exchange_taker_ratio['buyRatio']}%")
        print(f"Overall Sell Ratio: {exchange_taker_ratio['sellRatio']}%")
        print(f"Total Buy Volume: ${exchange_taker_ratio['buyVolUsd']:,.2f}")
        print(f"Total Sell Volume: ${exchange_taker_ratio['sellVolUsd']:,.2f}")

        # Plot Exchange Taker Buy/Sell Ratio
        plot_exchange_taker_buy_sell_ratio(
            exchange_taker_ratio, 
            "BTC Exchange Taker Buy/Sell Ratio (1h)", 
            "btc_exchange_taker_ratio_1h"
        )

        # Additional analysis
        df = pd.DataFrame(exchange_taker_ratio['list'])
        
        # Find exchange with highest buy ratio
        highest_buy = df.loc[df['buyRatio'].idxmax()]
        print(f"\nExchange with highest buy ratio: {highest_buy['exchange']} ({highest_buy['buyRatio']}%)")

        # Find exchange with highest sell ratio
        highest_sell = df.loc[df['sellRatio'].idxmax()]
        print(f"Exchange with highest sell ratio: {highest_sell['exchange']} ({highest_sell['sellRatio']}%)")

        # Calculate and print the correlation between buy ratio and volume
        correlation = df['buyRatio'].corr(df['buyVolUsd'])
        print(f"\nCorrelation between buy ratio and buy volume: {correlation:.2f}")
        
        
        # # Coins Markets Data
        # # Requires Standard Plan or higher
        # coins_markets_df = cg_api.coins_markets(exchanges="Binance,OKX,Bybit", page_size=100)
        # print("\nCoins Markets Data:")
        # print(f"Number of coins: {len(coins_markets_df)}")
        # print("\nTop 5 coins by market cap:")
        # print(coins_markets_df.sort_values('marketCap', ascending=False)[['symbol', 'price', 'marketCap', 'volUsd']].head())

        # # Plot Coins Markets Data
        # plot_coins_markets(coins_markets_df, "Crypto Markets Overview", "crypto_markets")

        # # Additional analysis
        # print("\nTop 5 coins by 24h volume:")
        # print(coins_markets_df.sort_values('volUsd', ascending=False)[['symbol', 'price', 'marketCap', 'volUsd']].head())

        # print("\nTop 5 coins by 24h price change:")
        # print(coins_markets_df.sort_values('priceChangePercent24h', ascending=False)[['symbol', 'price', 'priceChangePercent24h']].head())

        # print("\nTop 5 coins by open interest:")
        # print(coins_markets_df.sort_values('openInterest', ascending=False)[['symbol', 'price', 'openInterest', 'oiMarketCapRatio']].head())

        # # Calculate correlations
        # correlation_matrix = coins_markets_df[['price', 'marketCap', 'volUsd', 'openInterest', 'oiMarketCapRatio']].corr()
        # print("\nCorrelation matrix:")
        # print(correlation_matrix)

        # Pairs Markets Data
        pairs_markets_df = cg_api.pairs_markets(symbol="BTC")
        print("\nPairs Markets Data:")
        print(f"Number of trading pairs: {len(pairs_markets_df)}")
        print("\nTop 5 trading pairs by 24h volume:")
        print(pairs_markets_df.sort_values('volUsd', ascending=False)[['instrumentId', 'exName', 'price', 'volUsd']].head())

        # Plot Pairs Markets Data
        plot_pairs_markets(pairs_markets_df, "BTC Pairs Markets Overview", "btc_pairs_markets")

        # Additional analysis
        print("\nTop 5 trading pairs by open interest:")
        print(pairs_markets_df.sort_values('openInterest', ascending=False)[['instrumentId', 'exName', 'openInterest', 'oiChangePercent24h']].head())

        print("\nTop 5 trading pairs by 24h price change:")
        print(pairs_markets_df.sort_values('priceChangePercent24h', ascending=False)[['instrumentId', 'exName', 'price', 'priceChangePercent24h']].head())

        print("\nTop 5 trading pairs by funding rate:")
        print(pairs_markets_df.sort_values('fundingRate', ascending=False)[['instrumentId', 'exName', 'fundingRate', 'nextFundingTime']].head())

        # Calculate average metrics by exchange
        avg_by_exchange = pairs_markets_df.groupby('exName').agg({
            'volUsd': 'mean',
            'openInterest': 'mean',
            'fundingRate': 'mean',
            'oiVolRadio': 'mean'
        }).reset_index()

        print("\nAverage metrics by exchange:")
        print(avg_by_exchange)

        # Calculate correlations
        correlation_matrix = pairs_markets_df[['price', 'volUsd', 'openInterest', 'fundingRate', 'oiVolRadio']].corr()
        print("\nCorrelation matrix:")
        print(correlation_matrix)
        
        
        
        # Orderbook History
        orderbook_df = cg_api.orderbook_history(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nOrderbook History (BTCUSDT, last 7 days):")
        print(orderbook_df)

        # Plot Orderbook History
        plot_orderbook_history(orderbook_df, "BTCUSDT Orderbook History", "btcusdt_orderbook_history")

        # Additional analysis
        total_bids = orderbook_df['bidsUsd'].sum()
        total_asks = orderbook_df['asksUsd'].sum()
        print(f"\nTotal Bids (USD): ${total_bids:,.2f}")
        print(f"Total Asks (USD): ${total_asks:,.2f}")

        overall_bid_ask_ratio = total_bids / total_asks
        print(f"Overall Bid/Ask Ratio: {overall_bid_ask_ratio:.4f}")

        # Calculate percentage of time bids exceed asks
        bids_dominant = (orderbook_df['bidsUsd'] > orderbook_df['asksUsd']).mean() * 100
        print(f"Percentage of time bids exceeded asks: {bids_dominant:.2f}%")

        # Find the hour with the highest bids and asks
        max_bids_hour = orderbook_df.loc[orderbook_df['bidsUsd'].idxmax()]
        max_asks_hour = orderbook_df.loc[orderbook_df['asksUsd'].idxmax()]

        print(f"\nHour with highest bids: {max_bids_hour.name}")
        print(f"Bids (USD): ${max_bids_hour['bidsUsd']:,.2f}, Asks (USD): ${max_bids_hour['asksUsd']:,.2f}")

        print(f"\nHour with highest asks: {max_asks_hour.name}")
        print(f"Bids (USD): ${max_asks_hour['bidsUsd']:,.2f}, Asks (USD): ${max_asks_hour['asksUsd']:,.2f}")
        
        # 
        # Aggregated Orderbook History
        agg_orderbook_df = cg_api.aggregated_orderbook_history(
            exchanges="Binance,OKX,Bybit",
            symbol="BTC",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nAggregated Orderbook History (BTC, last 7 days):")
        print(agg_orderbook_df)

        # Plot Aggregated Orderbook History
        plot_aggregated_orderbook_history(agg_orderbook_df, "BTC Aggregated Orderbook History", "btc_aggregated_orderbook_history")

        # Additional analysis
        total_bids = agg_orderbook_df['bidsUsd'].sum()
        total_asks = agg_orderbook_df['asksUsd'].sum()
        print(f"\nTotal Aggregated Bids (USD): ${total_bids:,.2f}")
        print(f"Total Aggregated Asks (USD): ${total_asks:,.2f}")

        overall_bid_ask_ratio = total_bids / total_asks
        print(f"Overall Aggregated Bid/Ask Ratio: {overall_bid_ask_ratio:.4f}")

        # Calculate percentage of time bids exceed asks
        bids_dominant = (agg_orderbook_df['bidsUsd'] > agg_orderbook_df['asksUsd']).mean() * 100
        print(f"Percentage of time aggregated bids exceeded asks: {bids_dominant:.2f}%")

        # Find the hour with the highest bids and asks
        max_bids_hour = agg_orderbook_df.loc[agg_orderbook_df['bidsUsd'].idxmax()]
        max_asks_hour = agg_orderbook_df.loc[agg_orderbook_df['asksUsd'].idxmax()]

        print(f"\nHour with highest aggregated bids: {max_bids_hour.name}")
        print(f"Bids (USD): ${max_bids_hour['bidsUsd']:,.2f}, Asks (USD): ${max_bids_hour['asksUsd']:,.2f}")

        print(f"\nHour with highest aggregated asks: {max_asks_hour.name}")
        print(f"Bids (USD): ${max_asks_hour['bidsUsd']:,.2f}, Asks (USD): ${max_asks_hour['asksUsd']:,.2f}")

        # Compare single exchange (Binance) with aggregated data
        print("\nComparison between Binance and Aggregated Orderbook:")
        print(f"Binance Bid/Ask Ratio: {(orderbook_df['bidsUsd'].sum() / orderbook_df['asksUsd'].sum()):.4f}")
        print(f"Aggregated Bid/Ask Ratio: {overall_bid_ask_ratio:.4f}")

        correlation = orderbook_df['bidsUsd'].corr(agg_orderbook_df['bidsUsd'])
        print(f"Correlation between Binance and Aggregated Bids: {correlation:.4f}")

        correlation = orderbook_df['asksUsd'].corr(agg_orderbook_df['asksUsd'])
        print(f"Correlation between Binance and Aggregated Asks: {correlation:.4f}")
        
## 
## SPOT SECTION
        
        # Fetch supported coins for spot trading
        spot_supported_coins = cg_api.spot_supported_coins()
        print("\nSupported Coins for Spot Trading:")
        print(f"Total number of supported coins: {len(spot_supported_coins)}")
        print("First 10 supported coins:")
        print(", ".join(spot_supported_coins[:10]))

        # Optional: Save the full list to a file
        with open("images/spot_supported_coins.txt", "w") as f:
            f.write("\n".join(spot_supported_coins))
        print(f"Full list of supported coins saved to 'images/spot_supported_coins.txt'")

        # Fetch supported exchanges and pairs for spot trading
        spot_supported_exchanges = cg_api.spot_supported_exchange_pairs()
        print("\nSupported Exchanges and Pairs for Spot Trading:")
        print(f"Total number of supported exchanges: {len(spot_supported_exchanges)}")

        # Analyze the data
        total_pairs = sum(len(pairs) for pairs in spot_supported_exchanges.values())
        print(f"Total number of supported trading pairs across all exchanges: {total_pairs}")

        # Find the exchange with the most trading pairs
        max_exchange = max(spot_supported_exchanges, key=lambda x: len(spot_supported_exchanges[x]))
        print(f"Exchange with the most trading pairs: {max_exchange} ({len(spot_supported_exchanges[max_exchange])} pairs)")

        # Count the number of unique base assets across all exchanges
        unique_base_assets = set()
        for exchange_pairs in spot_supported_exchanges.values():
            unique_base_assets.update(pair['baseAsset'] for pair in exchange_pairs)
        print(f"Number of unique base assets across all exchanges: {len(unique_base_assets)}")

        # Print details for a specific exchange (e.g., Binance)
        if 'Binance' in spot_supported_exchanges:
            binance_pairs = spot_supported_exchanges['Binance']
            print(f"\nBinance supported pairs: {len(binance_pairs)}")
            print("First 5 Binance trading pairs:")
            for pair in binance_pairs[:5]:
                print(f"  {pair['instrumentId']} ({pair['baseAsset']}/{pair['quoteAsset']})")

        # Optional: Save the full data to a JSON file for further analysis
        import json
        with open("images/spot_supported_exchange_pairs.json", "w") as f:
            json.dump(spot_supported_exchanges, f, indent=2)
        print(f"Full data of supported exchanges and pairs saved to 'images/spot_supported_exchange_pairs.json'")
        
        
        # Fetch spot taker buy/sell volume history
        spot_taker_volume_df = cg_api.spot_taker_buy_sell_volume_history(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nSpot Taker Buy/Sell Volume History (BTCUSDT on Binance, last 7 days):")
        print(spot_taker_volume_df)

        # Plot Spot Taker Buy/Sell Volume and Ratio
        plot_taker_buy_sell_volume_ratio(spot_taker_volume_df, 
                                        "BTCUSDT Spot Taker Buy/Sell Volume on Binance", 
                                        "btcusdt_spot_taker_volume")

        # Additional analysis
        total_buy_volume = spot_taker_volume_df['buy'].sum()
        total_sell_volume = spot_taker_volume_df['sell'].sum()
        print(f"\nTotal Spot Buy Volume: {total_buy_volume:,.2f}")
        print(f"Total Spot Sell Volume: {total_sell_volume:,.2f}")

        overall_buy_sell_ratio = total_buy_volume / total_sell_volume
        print(f"Overall Spot Buy/Sell Ratio: {overall_buy_sell_ratio:.4f}")

        # Calculate percentage of time buy volume exceeds sell volume
        buy_dominant = (spot_taker_volume_df['buy'] > spot_taker_volume_df['sell']).mean() * 100
        print(f"Percentage of time spot buy volume exceeded sell volume: {buy_dominant:.2f}%")

        # Find the hour with the highest buy and sell volumes
        max_buy_hour = spot_taker_volume_df.loc[spot_taker_volume_df['buy'].idxmax()]
        max_sell_hour = spot_taker_volume_df.loc[spot_taker_volume_df['sell'].idxmax()]

        print(f"\nHour with highest spot buy volume: {max_buy_hour.name}")
        print(f"Buy volume: {max_buy_hour['buy']:,.2f}, Sell volume: {max_buy_hour['sell']:,.2f}")

        print(f"\nHour with highest spot sell volume: {max_sell_hour.name}")
        print(f"Buy volume: {max_sell_hour['buy']:,.2f}, Sell volume: {max_sell_hour['sell']:,.2f}")

        # Calculate and display average buy/sell ratio
        avg_ratio = (spot_taker_volume_df['buy'] / spot_taker_volume_df['sell']).mean()
        print(f"\nAverage Spot Buy/Sell Ratio: {avg_ratio:.4f}")

        # Find and display the highest and lowest ratios
        spot_taker_volume_df['ratio'] = spot_taker_volume_df['buy'] / spot_taker_volume_df['sell']
        max_ratio = spot_taker_volume_df['ratio'].max()
        min_ratio = spot_taker_volume_df['ratio'].min()
        print(f"Highest Spot Buy/Sell Ratio: {max_ratio:.4f}")
        print(f"Lowest Spot Buy/Sell Ratio: {min_ratio:.4f}")
        
        # Plot Spot Taker Buy/Sell Volume
        plot_spot_taker_buy_sell_volume(spot_taker_volume_df, 
                                "BTCUSDT Spot Taker Buy/Sell Volume on Binance", 
                                "btcusdt_spot_taker_volume")
        
        
        # Fetch spot aggregated taker buy/sell volume history
        spot_agg_taker_volume_df = cg_api.spot_aggregated_taker_buy_sell_volume_history(
            exchanges="Binance,OKX,Bybit",
            symbol="BTC",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nSpot Aggregated Taker Buy/Sell Volume History (BTC on Binance,OKX,Bybit, last 7 days):")
        print(spot_agg_taker_volume_df)

        # Plot Spot Aggregated Taker Buy/Sell Volume
        plot_spot_aggregated_taker_buy_sell_volume(spot_agg_taker_volume_df, 
                                                "BTC Spot Aggregated Taker Buy/Sell Volume", 
                                                "btc_spot_aggregated_taker_volume")

        # Additional analysis
        total_agg_buy_volume = spot_agg_taker_volume_df['buy'].sum()
        total_agg_sell_volume = spot_agg_taker_volume_df['sell'].sum()
        print(f"\nTotal Aggregated Spot Buy Volume: {total_agg_buy_volume:,.2f}")
        print(f"Total Aggregated Spot Sell Volume: {total_agg_sell_volume:,.2f}")

        overall_agg_buy_sell_ratio = total_agg_buy_volume / total_agg_sell_volume
        print(f"Overall Aggregated Spot Buy/Sell Ratio: {overall_agg_buy_sell_ratio:.4f}")

        # Calculate percentage of time buy volume exceeds sell volume
        agg_buy_dominant = (spot_agg_taker_volume_df['buy'] > spot_agg_taker_volume_df['sell']).mean() * 100
        print(f"Percentage of time aggregated spot buy volume exceeded sell volume: {agg_buy_dominant:.2f}%")

        # Find the hour with the highest buy and sell volumes
        max_agg_buy_hour = spot_agg_taker_volume_df.loc[spot_agg_taker_volume_df['buy'].idxmax()]
        max_agg_sell_hour = spot_agg_taker_volume_df.loc[spot_agg_taker_volume_df['sell'].idxmax()]

        print(f"\nHour with highest aggregated spot buy volume: {max_agg_buy_hour.name}")
        print(f"Buy volume: {max_agg_buy_hour['buy']:,.2f}, Sell volume: {max_agg_buy_hour['sell']:,.2f}")

        print(f"\nHour with highest aggregated spot sell volume: {max_agg_sell_hour.name}")
        print(f"Buy volume: {max_agg_sell_hour['buy']:,.2f}, Sell volume: {max_agg_sell_hour['sell']:,.2f}")

        # Calculate and display average buy/sell ratio
        avg_agg_ratio = (spot_agg_taker_volume_df['buy'] / spot_agg_taker_volume_df['sell']).mean()
        print(f"\nAverage Aggregated Spot Buy/Sell Ratio: {avg_agg_ratio:.4f}")

        # Find and display the highest and lowest ratios
        spot_agg_taker_volume_df['ratio'] = spot_agg_taker_volume_df['buy'] / spot_agg_taker_volume_df['sell']
        max_agg_ratio = spot_agg_taker_volume_df['ratio'].max()
        min_agg_ratio = spot_agg_taker_volume_df['ratio'].min()
        print(f"Highest Aggregated Spot Buy/Sell Ratio: {max_agg_ratio:.4f}")
        print(f"Lowest Aggregated Spot Buy/Sell Ratio: {min_agg_ratio:.4f}")
        
        # Verify timestamp handling for spot taker volume data
        print("\nSpot Taker Buy/Sell Volume Data Range:")
        print(f"Start: {spot_taker_volume_df.index.min()}")
        print(f"End: {spot_taker_volume_df.index.max()}")

        # Verify timestamp handling for spot aggregated taker volume data
        print("\nSpot Aggregated Taker Buy/Sell Volume Data Range:")
        print(f"Start: {spot_agg_taker_volume_df.index.min()}")
        print(f"End: {spot_agg_taker_volume_df.index.max()}")
        
    # Spot OrderBook History
        spot_orderbook_df = cg_api.spot_orderbook_history(
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nSpot OrderBook History (BTCUSDT on Binance, last 7 days):")
        print(spot_orderbook_df)

        # Plot Spot OrderBook History
        plot_spot_orderbook_history(spot_orderbook_df, 
                                    "BTCUSDT Spot OrderBook History on Binance", 
                                    "btcusdt_spot_orderbook_history")

        # Additional analysis
        total_bids = spot_orderbook_df['bidsUsd'].sum()
        total_asks = spot_orderbook_df['asksUsd'].sum()
        print(f"\nTotal Spot Bids (USD): ${total_bids:,.2f}")
        print(f"Total Spot Asks (USD): ${total_asks:,.2f}")

        overall_bid_ask_ratio = total_bids / total_asks
        print(f"Overall Spot Bid/Ask Ratio: {overall_bid_ask_ratio:.4f}")

        # Calculate percentage of time bids exceed asks
        bids_dominant = (spot_orderbook_df['bidsUsd'] > spot_orderbook_df['asksUsd']).mean() * 100
        print(f"Percentage of time spot bids exceeded asks: {bids_dominant:.2f}%")

        # Find the hour with the highest bids and asks
        max_bids_hour = spot_orderbook_df.loc[spot_orderbook_df['bidsUsd'].idxmax()]
        max_asks_hour = spot_orderbook_df.loc[spot_orderbook_df['asksUsd'].idxmax()]

        print(f"\nHour with highest spot bids: {max_bids_hour.name}")
        print(f"Bids (USD): ${max_bids_hour['bidsUsd']:,.2f}, Asks (USD): ${max_bids_hour['asksUsd']:,.2f}")

        print(f"\nHour with highest spot asks: {max_asks_hour.name}")
        print(f"Bids (USD): ${max_asks_hour['bidsUsd']:,.2f}, Asks (USD): ${max_asks_hour['asksUsd']:,.2f}")
        
    # Spot Aggregated OrderBook History
        spot_agg_orderbook_df = cg_api.spot_aggregated_orderbook_history(
            exchanges="Binance,OKX,Bybit",
            symbol="BTC",
            interval="1h",
            limit=168  # Last 7 days of hourly data
        )
        print("\nSpot Aggregated OrderBook History (BTC on Binance,OKX,Bybit, last 7 days):")
        print(spot_agg_orderbook_df)

        # Plot Spot Aggregated OrderBook History
        plot_spot_orderbook_history(spot_agg_orderbook_df, 
                                    "BTC Spot Aggregated OrderBook History", 
                                    "btc_spot_aggregated_orderbook_history",
                                    aggregated=True)

        # Additional analysis
        total_agg_bids = spot_agg_orderbook_df['bidsUsd'].sum()
        total_agg_asks = spot_agg_orderbook_df['asksUsd'].sum()
        print(f"\nTotal Aggregated Spot Bids (USD): ${total_agg_bids:,.2f}")
        print(f"Total Aggregated Spot Asks (USD): ${total_agg_asks:,.2f}")

        overall_agg_bid_ask_ratio = total_agg_bids / total_agg_asks
        print(f"Overall Aggregated Spot Bid/Ask Ratio: {overall_agg_bid_ask_ratio:.4f}")

        # Calculate percentage of time bids exceed asks
        agg_bids_dominant = (spot_agg_orderbook_df['bidsUsd'] > spot_agg_orderbook_df['asksUsd']).mean() * 100
        print(f"Percentage of time aggregated spot bids exceeded asks: {agg_bids_dominant:.2f}%")

        # Find the hour with the highest bids and asks
        max_agg_bids_hour = spot_agg_orderbook_df.loc[spot_agg_orderbook_df['bidsUsd'].idxmax()]
        max_agg_asks_hour = spot_agg_orderbook_df.loc[spot_agg_orderbook_df['asksUsd'].idxmax()]

        print(f"\nHour with highest aggregated spot bids: {max_agg_bids_hour.name}")
        print(f"Bids (USD): ${max_agg_bids_hour['bidsUsd']:,.2f}, Asks (USD): ${max_agg_bids_hour['asksUsd']:,.2f}")

        print(f"\nHour with highest aggregated spot asks: {max_agg_asks_hour.name}")
        print(f"Bids (USD): ${max_agg_asks_hour['bidsUsd']:,.2f}, Asks (USD): ${max_agg_asks_hour['asksUsd']:,.2f}")

        # Compare single exchange (Binance) with aggregated data
        print("\nComparison between Binance and Aggregated Spot Orderbook:")
        print(f"Binance Bid/Ask Ratio: {(spot_orderbook_df['bidsUsd'].sum() / spot_orderbook_df['asksUsd'].sum()):.4f}")
        print(f"Aggregated Bid/Ask Ratio: {overall_agg_bid_ask_ratio:.4f}")

        correlation = spot_orderbook_df['bidsUsd'].corr(spot_agg_orderbook_df['bidsUsd'])
        print(f"Correlation between Binance and Aggregated Bids: {correlation:.4f}")

        correlation = spot_orderbook_df['asksUsd'].corr(spot_agg_orderbook_df['asksUsd'])
        print(f"Correlation between Binance and Aggregated Asks: {correlation:.4f}")
        
    # Spot Pairs Markets
        spot_pairs_markets_df = cg_api.spot_pairs_markets(symbol="BTC")
        print("\nSpot Pairs Markets Data (BTC):")
        print(spot_pairs_markets_df)

        # Plot Spot Pairs Markets Data
        plot_spot_pairs_markets(spot_pairs_markets_df, "BTC Spot Pairs Markets Overview", "btc_spot_pairs_markets")

        # Additional analysis
        print("\nTop 5 trading pairs by 24h volume:")
        print(spot_pairs_markets_df.sort_values('volUsd24h', ascending=False)[['symbol', 'exName', 'price', 'volUsd24h']].head())

        print("\nTop 5 trading pairs by 24h price change:")
        print(spot_pairs_markets_df.sort_values('priceChangePercent24h', ascending=False)[['symbol', 'exName', 'price', 'priceChangePercent24h']].head())

        # Calculate average metrics by exchange
        avg_by_exchange = spot_pairs_markets_df.groupby('exName').agg({
            'volUsd24h': 'mean',
            'priceChangePercent24h': 'mean',
            'volUsdChangePercent24h': 'mean'
        }).reset_index()

        print("\nAverage metrics by exchange:")
        print(avg_by_exchange)

        # Calculate correlations
        correlation_matrix = spot_pairs_markets_df[['price', 'volUsd24h', 'priceChangePercent24h', 'volUsdChangePercent24h']].corr()
        print("\nCorrelation matrix:")
        print(correlation_matrix)
##
## INDICATIR SECTION
##        
        # Bitcoin Bubble Index
        bitcoin_bubble_df = cg_api.bitcoin_bubble_index()
        print("\nBitcoin Bubble Index Data:")
        print(bitcoin_bubble_df.tail())

        # Plot Bitcoin Bubble Index
        plot_bitcoin_bubble_index(bitcoin_bubble_df, "Bitcoin Bubble Index", "bitcoin_bubble_index")

        # Additional analysis
        current_index = bitcoin_bubble_df['index'].iloc[-1]
        current_price = bitcoin_bubble_df['price'].iloc[-1]
        print(f"\nCurrent Bitcoin Bubble Index: {current_index:.2f}")
        print(f"Current Bitcoin Price: ${current_price:.2f}")

        # Calculate correlation between index and price
        correlation = bitcoin_bubble_df['index'].corr(bitcoin_bubble_df['price'])
        print(f"\nCorrelation between Bitcoin Bubble Index and Price: {correlation:.2f}")

        # Find highest and lowest index values
        max_index = bitcoin_bubble_df['index'].max()
        min_index = bitcoin_bubble_df['index'].min()
        max_index_date = bitcoin_bubble_df['index'].idxmax()
        min_index_date = bitcoin_bubble_df['index'].idxmin()

        print(f"\nHighest Bitcoin Bubble Index: {max_index:.2f} on {max_index_date.date()}")
        print(f"Lowest Bitcoin Bubble Index: {min_index:.2f} on {min_index_date.date()}")

        # Calculate average index for different price ranges
        bitcoin_bubble_df['price_range'] = pd.cut(bitcoin_bubble_df['price'], bins=[0, 1000, 10000, 50000, float('inf')], labels=['0-1k', '1k-10k', '10k-50k', '50k+'])
        
        avg_index_by_price_range = bitcoin_bubble_df.groupby('price_range', observed=True)['index'].mean()
        print("\nAverage Bitcoin Bubble Index by Price Range:")
        print(avg_index_by_price_range)
        
        # AHR999 Index
        ahr999_df = cg_api.ahr999_index()
        print("\nAHR999 Index Data:")
        print(ahr999_df.tail())

        # Plot AHR999 Index
        plot_ahr999_index(ahr999_df, "AHR999 Index", "ahr999_index")

        # Additional analysis
        current_ahr999 = ahr999_df['ahr999'].iloc[-1]
        current_price = float(ahr999_df['value'].iloc[-1])
        print(f"\nCurrent AHR999 Index: {current_ahr999:.2f}")
        print(f"Current Bitcoin Price: ${current_price:.2f}")

        # Calculate correlation between AHR999 and price
        ahr999_df['price'] = ahr999_df['value'].astype(float)
        correlation = ahr999_df['ahr999'].corr(ahr999_df['price'])
        print(f"\nCorrelation between AHR999 Index and Price: {correlation:.2f}")

        # Find highest and lowest AHR999 values
        max_ahr999 = ahr999_df['ahr999'].max()
        min_ahr999 = ahr999_df['ahr999'].min()
        max_ahr999_date = ahr999_df['ahr999'].idxmax()
        min_ahr999_date = ahr999_df['ahr999'].idxmin()

        print(f"\nHighest AHR999 Index: {max_ahr999:.2f} on {max_ahr999_date.date()}")
        print(f"Lowest AHR999 Index: {min_ahr999:.2f} on {min_ahr999_date.date()}")

        # Calculate average AHR999 for different price ranges
        ahr999_df['price_range'] = pd.cut(ahr999_df['price'], bins=[0, 1000, 10000, 50000, float('inf')], labels=['0-1k', '1k-10k', '10k-50k', '50k+'])
        avg_ahr999_by_price_range = ahr999_df.groupby('price_range', observed=True)['ahr999'].mean()
        
        print("\nAverage AHR999 Index by Price Range:")
        print(avg_ahr999_by_price_range)
        
        # Two Year MA Multiplier
        two_year_ma_df = cg_api.two_year_ma_multiplier()
        print("\nTwo Year MA Multiplier Data:")
        print(two_year_ma_df.tail())

        # Plot Two Year MA Multiplier
        plot_two_year_ma_multiplier(two_year_ma_df, "Two Year MA Multiplier", "two_year_ma_multiplier")

        # Additional analysis
        current_price = two_year_ma_df['price'].iloc[-1]
        current_ma730 = two_year_ma_df['mA730'].iloc[-1]
        current_ma730mu5 = two_year_ma_df['mA730Mu5'].iloc[-1]
        print(f"\nCurrent Bitcoin Price: ${current_price:.2f}")
        print(f"Current 2Y MA: ${current_ma730:.2f}")
        print(f"Current 2Y MA x5: ${current_ma730mu5:.2f}")

        # Calculate price position relative to 2Y MA and 2Y MA x5
        price_to_ma_ratio = current_price / current_ma730
        price_to_ma5_ratio = current_price / current_ma730mu5
        print(f"\nPrice to 2Y MA ratio: {price_to_ma_ratio:.2f}")
        print(f"Price to 2Y MA x5 ratio: {price_to_ma5_ratio:.2f}")

        # Find highest and lowest price relative to 2Y MA
        two_year_ma_df['price_to_ma_ratio'] = two_year_ma_df['price'] / two_year_ma_df['mA730']
        max_ratio = two_year_ma_df['price_to_ma_ratio'].max()
        min_ratio = two_year_ma_df['price_to_ma_ratio'].min()
        max_ratio_date = two_year_ma_df['price_to_ma_ratio'].idxmax()
        min_ratio_date = two_year_ma_df['price_to_ma_ratio'].idxmin()

        print(f"\nHighest Price to 2Y MA ratio: {max_ratio:.2f} on {max_ratio_date.date()}")
        print(f"Lowest Price to 2Y MA ratio: {min_ratio:.2f} on {min_ratio_date.date()}")

        # Calculate percentage of time price is between 2Y MA and 2Y MA x5
        between_ma_and_ma5 = ((two_year_ma_df['price'] >= two_year_ma_df['mA730']) & 
                            (two_year_ma_df['price'] <= two_year_ma_df['mA730Mu5']))
        percentage_between = between_ma_and_ma5.mean() * 100
        print(f"\nPercentage of time price is between 2Y MA and 2Y MA x5: {percentage_between:.2f}%")
        
        # 200-Week Moving Average Heatmap
        ma_200w_df = cg_api.two_hundred_week_moving_avg_heatmap()
        print("\n200-Week Moving Average Heatmap Data:")
        print(ma_200w_df.tail())
        plot_200_week_ma_heatmap(ma_200w_df, "200-Week Moving Average Heatmap", "200_week_ma_heatmap")

        # Puell Multiple
        puell_df = cg_api.puell_multiple()
        print("\nPuell Multiple Data:")
        print(puell_df.tail())
        plot_puell_multiple(puell_df, "Puell Multiple", "puell_multiple")

        # Stock-to-Flow Model
        stf_df = cg_api.stock_to_flow()
        print("\nStock-to-Flow Model Data:")
        print(stf_df.tail())
        plot_stock_to_flow(stf_df, "Stock-to-Flow Model", "stock_to_flow")

        # Pi Cycle Top Indicator
        pi_df = cg_api.pi_cycle_top_indicator()
        print("\nPi Cycle Top Indicator Data:")
        print(pi_df.tail())
        plot_pi_cycle_top_indicator(pi_df, "Pi Cycle Top Indicator", "pi_cycle_top")

        # Additional analysis for each indicator
        print("\n--- 200-Week Moving Average Heatmap Analysis ---")
        current_price = float(ma_200w_df['price'].iloc[-1])
        current_ma = float(ma_200w_df['mA1440'].iloc[-1])
        price_to_ma_ratio = current_price / current_ma
        print(f"Current Price: ${current_price:.2f}")
        print(f"Current 200-Week MA: ${current_ma:.2f}")
        print(f"Price to 200-Week MA Ratio: {price_to_ma_ratio:.2f}")

        print("\n--- Puell Multiple Analysis ---")
        current_puell = float(puell_df['puellMultiple'].iloc[-1])
        print(f"Current Puell Multiple: {current_puell:.2f}")
        print(f"Max Puell Multiple: {puell_df['puellMultiple'].astype(float).max():.2f}")
        print(f"Min Puell Multiple: {puell_df['puellMultiple'].astype(float).min():.2f}")

       
        print("\n--- Stock-to-Flow Model Analysis ---")
        # Filter out NaN values
        stf_df = stf_df.dropna(subset=['price', 'stockFlow365dAverage'])

        current_stf = float(stf_df['stockFlow365dAverage'].iloc[-1])
        current_price = float(stf_df['price'].iloc[-1])
        print(f"Current Stock-to-Flow: {current_stf:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Days until next halving: {stf_df['nextHalving'].iloc[-1]}")

        # Add this section to check the date range
        print(f"Data start date: {stf_df.index.min()}")
        print(f"Data end date: {stf_df.index.max()}")
        print(f"Total data points: {len(stf_df)}")

        # Calculate correlation between price and Stock-to-Flow
        correlation = stf_df['price'].astype(float).corr(stf_df['stockFlow365dAverage'].astype(float))
        print(f"Correlation between Price and Stock-to-Flow: {correlation:.2f}")

        print("\n--- Pi Cycle Top Indicator Analysis ---")
        current_price = float(pi_df['price'].iloc[-1])
        current_ma110 = float(pi_df['ma110'].iloc[-1])
        current_ma350mu2 = float(pi_df['ma350Mu2'].iloc[-1])
        print(f"Current Price: ${current_price:.2f}")
        print(f"Current 110 DMA: ${current_ma110:.2f}")
        print(f"Current 350 DMA x2: ${current_ma350mu2:.2f}")
        
        # Golden Ratio Multiplier
        grm_df = cg_api.golden_ratio_multiplier()
        print("\nGolden Ratio Multiplier Data:")
        print(grm_df.tail())
        plot_golden_ratio_multiplier(grm_df, "Golden Ratio Multiplier", "golden_ratio_multiplier")

        # Bitcoin Profitable Days
        bpd_df = cg_api.bitcoin_profitable_days()
        print("\nBitcoin Profitable Days Data:")
        print(bpd_df.tail())
        plot_bitcoin_profitable_days(bpd_df, "Bitcoin Profitable Days", "bitcoin_profitable_days")

        # Bitcoin Rainbow Chart
        brc_df = cg_api.bitcoin_rainbow_chart()
        print("\nBitcoin Rainbow Chart Data:")
        print(brc_df.tail())
        plot_bitcoin_rainbow_chart(brc_df, "Bitcoin Rainbow Chart", "bitcoin_rainbow_chart")

        # Crypto Fear & Greed Index
        fgi_df = cg_api.fear_greed_index()
        print("\nCrypto Fear & Greed Index Data:")
        print(fgi_df.tail())
        plot_fear_greed_index(fgi_df, "Crypto Fear & Greed Index", "fear_greed_index")

        # Additional analysis
        current_fgi = fgi_df['values'].iloc[-1]
        print(f"\nCurrent Fear & Greed Index: {current_fgi:.2f}")
        print(f"Minimum Fear & Greed Index: {fgi_df['values'].min():.2f}")
        print(f"Maximum Fear & Greed Index: {fgi_df['values'].max():.2f}")
        print(f"Average Fear & Greed Index: {fgi_df['values'].mean():.2f}")

        if 'price' in fgi_df.columns:
            # Remove NaN values before calculating correlation
            valid_data = fgi_df.dropna(subset=['values', 'price'])
            correlation = valid_data['values'].corr(valid_data['price'])
            print(f"Correlation between Fear & Greed Index and Price: {correlation:.2f}")
        else:
            print("Price data not available for correlation analysis")

                
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