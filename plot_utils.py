import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection

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



def plot_long_short_ratio(global_df, top_df, title, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Global Account Ratio
    sns.lineplot(data=global_df, x=global_df.index, y='longAccount', label='Long Account', ax=ax1)
    sns.lineplot(data=global_df, x=global_df.index, y='shortAccount', label='Short Account', ax=ax1)
    ax1.set_title(f"{title} - Global Account Ratio")
    ax1.set_ylabel("Account Percentage")
    ax1.legend()

    # Top Account Ratio
    sns.lineplot(data=top_df, x=top_df.index, y='longAccount', label='Long Account', ax=ax2)
    sns.lineplot(data=top_df, x=top_df.index, y='shortAccount', label='Short Account', ax=ax2)
    ax2.set_title(f"{title} - Top Account Ratio")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Account Percentage")
    ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}_accounts.png")
    plt.close()

    # Long/Short Ratio Comparison
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=global_df, x=global_df.index, y='longShortRatio', label='Global')
    sns.lineplot(data=top_df, x=top_df.index, y='longShortRatio', label='Top Accounts')
    plt.title(f"{title} - Long/Short Ratio Comparison")
    plt.xlabel("Date")
    plt.ylabel("Long/Short Ratio")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}_ratio_comparison.png")
    plt.close()
    


    
def plot_liquidation_heatmap_v2(heatmap_data, title, filename, range="3d"):
    price_levels = heatmap_data['price_levels']
    liquidations = heatmap_data['liquidations']
    ohlc = heatmap_data['ohlc']
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 20, figure=fig)
    
    # Create axes for colorbar and main plot
    cbar_ax = fig.add_subplot(gs[0, 0])  # Leftmost column for colorbar
    ax = fig.add_subplot(gs[0, 1:])  # Rest for the main plot
    
    # Create heatmap data
    heatmap = np.zeros((len(price_levels), len(ohlc)))
    for _, liq in liquidations.iterrows():
        heatmap[int(liq['price_index'])][int(liq['time_index'])] = liq['liquidation_value']

    # Set fixed vmax for 1y scale
    vmax = 5.88e9 if range == "1y" else liquidations['liquidation_value'].max()
    
    # Plot heatmap
    extent = [mdates.date2num(ohlc.index[0]), mdates.date2num(ohlc.index[-1]), 
              price_levels['price'].min(), price_levels['price'].max()]
    
    colors = ['#3a0ca3', '#4361ee', '#4cc9f0', '#7bf1a8', '#ccff33']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    norm = LogNorm(vmin=1e4, vmax=vmax)
    im = ax.imshow(heatmap, aspect='auto', extent=extent, origin='lower', 
                   cmap=cmap, norm=norm, interpolation='nearest')

   
    
    # Customize y-axis (price)
    ax.set_ylabel('Bitcoin Price (USDT)', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Customize x-axis (date)
    ax.set_xlabel('Date', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Add colorbar on the far left
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Liquidation Value', fontsize=12, labelpad=15)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.2e}"))
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.tick_left()

    # Set title
    ax.set_title(title, fontsize=16, pad=20)

    # plt.tight_layout()
    plt.savefig(f"images/{filename}_{range}.png", dpi=300, bbox_inches='tight')
    plt.close()
    price_levels = heatmap_data['price_levels']
    liquidations = heatmap_data['liquidations']
    ohlc = heatmap_data['ohlc']
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create heatmap data
    heatmap = np.zeros((len(price_levels), len(ohlc)))
    for _, liq in liquidations.iterrows():
        heatmap[int(liq['price_index'])][int(liq['time_index'])] = liq['liquidation_value']

    # Plot heatmap
    extent = [mdates.date2num(ohlc.index[0]), mdates.date2num(ohlc.index[-1]), 
            0, len(price_levels) - 1]
    
    colors = ['#3a0ca3', '#4361ee', '#4cc9f0', '#7bf1a8', '#ccff33']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
    im = ax.imshow(heatmap, aspect='auto', extent=extent, origin='lower', 
                cmap=cmap, interpolation='nearest')

    # Set y-axis to show Bitcoin prices
    price_ticks = np.linspace(0, len(price_levels) - 1, num=10, dtype=int)
    ax.set_yticks(price_ticks)
    ax.set_yticklabels([f"{price_levels['price'][i]:,.0f}" for i in price_ticks])

    # Plot OHLC data
    ax_price = ax.twinx()
    ax_price.plot(ohlc.index, ohlc['close'], color='black', alpha=0.75, linewidth=1)
    ax_price.fill_between(ohlc.index, ohlc['low'], ohlc['high'], color='gray', alpha=0.2)
    ax_price.set_ylim(price_levels['price'].min(), price_levels['price'].max())
    ax_price.set_ylabel('Bitcoin Price (USDT)', fontsize=12)

     # Create a color array based on price changes
    colors = ['green']
    close_prices = ohlc['close'].values
    dates = mdates.date2num(ohlc.index)
    points = np.array([dates, close_prices]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors, linewidth=1.5, alpha=0.8)
    ax_price.add_collection(lc)
    
    
    # Customize subplot
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Bitcoin Price (USDT)', fontsize=12)

    # Customize x-axis
    ax.set_xlabel('Date', fontsize=12)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ## Add colorbar
    #cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    #cbar.set_label('Liquidation Value', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"images/{filename}_{range}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_aggregated_taker_buy_sell_ratio(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['longShortRatio'], label='Long/Short Ratio')
    plt.axhline(y=1, color='r', linestyle='--', label='Equilibrium')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Long/Short Ratio")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()
    
def plot_aggregated_taker_buy_sell_volume(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['buy'], label='Buy Volume', color='green')
    plt.plot(df.index, df['sell'], label='Sell Volume', color='red')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()
    
def plot_taker_buy_sell_volume_ratio(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['buy'], label='Buy Volume', color='green')
    plt.plot(df.index, df['sell'], label='Sell Volume', color='red')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

    # Plot ratio
    plt.figure(figsize=(12, 6))
    ratio = df['buy'] / df['sell']
    plt.plot(df.index, ratio, label='Buy/Sell Ratio', color='blue')
    plt.axhline(y=1, color='r', linestyle='--', label='Equilibrium')
    plt.title(f"{title} - Buy/Sell Ratio")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}_ratio.png")
    plt.close()
    
def plot_exchange_taker_buy_sell_ratio(data: dict, title: str, filename: str):
    """
    Plot the exchange taker buy/sell ratio data.

    Args:
        data: Dictionary containing the API response data
        title: Title for the plot
        filename: Filename to save the plot
    """
    df = pd.DataFrame(data['list'])
    df = df.sort_values('buyRatio', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df['exchange'], df['buyRatio'], label='Buy Ratio', color='green', alpha=0.7)
    plt.bar(df['exchange'], df['sellRatio'], bottom=df['buyRatio'], label='Sell Ratio', color='red', alpha=0.7)

    plt.title(title)
    plt.xlabel("Exchange")
    plt.ylabel("Ratio (%)")
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

    # Create a second plot for volume
    plt.figure(figsize=(12, 6))
    plt.bar(df['exchange'], df['buyVolUsd'], label='Buy Volume', color='green', alpha=0.7)
    plt.bar(df['exchange'], df['sellVolUsd'], bottom=df['buyVolUsd'], label='Sell Volume', color='red', alpha=0.7)

    plt.title(f"{title} - Volume")
    plt.xlabel("Exchange")
    plt.ylabel("Volume (USD)")
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"images/{filename}_volume.png")
    plt.close()
    
def plot_coins_markets(df: pd.DataFrame, title_prefix: str, filename_prefix: str):
    """
    Create multiple plots for coins market data.

    Args:
        df: DataFrame containing coins market data
        title_prefix: Prefix for plot titles
        filename_prefix: Prefix for saved plot filenames
    """
    # Market Cap vs. Volume Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='marketCap', y='volUsd', hue='symbol', size='openInterest', sizes=(20, 500), alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"{title_prefix} - Market Cap vs. Volume")
    plt.xlabel("Market Cap (USD)")
    plt.ylabel("24h Volume (USD)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}_market_cap_vs_volume.png")
    plt.close()

    # Price Change Heatmap
    price_change_columns = [col for col in df.columns if col.startswith('priceChangePercent')]
    price_change_df = df[['symbol'] + price_change_columns].set_index('symbol')
    plt.figure(figsize=(12, 10))
    sns.heatmap(price_change_df, cmap='RdYlGn', center=0, annot=True, fmt='.2f')
    plt.title(f"{title_prefix} - Price Change Heatmap")
    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}_price_change_heatmap.png")
    plt.close()

    # Long/Short Ratio Bar Plot
    ls_columns = [col for col in df.columns if col.startswith('ls') and col != 'ls']
    ls_df = df[['symbol'] + ls_columns].set_index('symbol')
    ls_df = ls_df.sort_values('ls24h', ascending=False)
    plt.figure(figsize=(12, 8))
    ls_df.plot(kind='bar', width=0.8)
    plt.title(f"{title_prefix} - Long/Short Ratios")
    plt.xlabel("Symbol")
    plt.ylabel("Long/Short Ratio")
    plt.legend(title="Time Frame", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}_long_short_ratios.png")
    plt.close()
    
def plot_pairs_markets(df: pd.DataFrame, title_prefix: str, filename_prefix: str):
    """
    Create multiple plots for pairs market data.

    Args:
        df: DataFrame containing pairs market data
        title_prefix: Prefix for plot titles
        filename_prefix: Prefix for saved plot filenames
    """
    # Trading Volume by Exchange
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='exName', y='volUsd', hue='symbol')
    plt.title(f"{title_prefix} - Trading Volume by Exchange")
    plt.xlabel("Exchange")
    plt.ylabel("24h Volume (USD)")
    plt.xticks(rotation=45)
    plt.legend(title="Symbol", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}_volume_by_exchange.png")
    plt.close()

    # Open Interest by Exchange
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='exName', y='openInterest', hue='symbol')
    plt.title(f"{title_prefix} - Open Interest by Exchange")
    plt.xlabel("Exchange")
    plt.ylabel("Open Interest (USD)")
    plt.xticks(rotation=45)
    plt.legend(title="Symbol", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}_open_interest_by_exchange.png")
    plt.close()

    # Funding Rates Comparison
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='exName', y='fundingRate', hue='symbol')
    plt.title(f"{title_prefix} - Funding Rates Comparison")
    plt.xlabel("Exchange")
    plt.ylabel("Funding Rate")
    plt.xticks(rotation=45)
    plt.legend(title="Symbol", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}_funding_rates_comparison.png")
    plt.close()

    # Long/Short Volume Ratio
    df['longShortRatio'] = df['longVolUsd'] / df['shortVolUsd']
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='exName', y='longShortRatio', hue='symbol')
    plt.title(f"{title_prefix} - Long/Short Volume Ratio")
    plt.xlabel("Exchange")
    plt.ylabel("Long/Short Ratio")
    plt.xticks(rotation=45)
    plt.legend(title="Symbol", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}_long_short_ratio.png")
    plt.close()
    
def plot_orderbook_history(df: pd.DataFrame, title: str, filename: str):
    """
    Plot orderbook history data.

    Args:
        df: DataFrame containing orderbook history data
        title: Title for the plot
        filename: Filename to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot bids and asks
    plt.plot(df.index, df['bidsUsd'], label='Bids', color='green')
    plt.plot(df.index, df['asksUsd'], label='Asks', color='red')
    
    # Plot the difference between bids and asks
    plt.fill_between(df.index, df['bidsUsd'], df['asksUsd'], alpha=0.3, 
                     where=(df['bidsUsd'] > df['asksUsd']), facecolor='green', interpolate=True)
    plt.fill_between(df.index, df['bidsUsd'], df['asksUsd'], alpha=0.3, 
                     where=(df['asksUsd'] > df['bidsUsd']), facecolor='red', interpolate=True)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("USD Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

    # Plot bid/ask ratio
    plt.figure(figsize=(12, 6))
    ratio = df['bidsUsd'] / df['asksUsd']
    plt.plot(df.index, ratio, label='Bid/Ask Ratio', color='blue')
    plt.axhline(y=1, color='r', linestyle='--', label='Equilibrium')
    plt.title(f"{title} - Bid/Ask Ratio")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}_ratio.png")
    plt.close()
    
def plot_aggregated_orderbook_history(df: pd.DataFrame, title: str, filename: str):
    """
    Plot aggregated orderbook history data.

    Args:
        df: DataFrame containing aggregated orderbook history data
        title: Title for the plot
        filename: Filename to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot bids and asks
    plt.plot(df.index, df['bidsUsd'], label='Bids', color='green')
    plt.plot(df.index, df['asksUsd'], label='Asks', color='red')
    
    # Plot the difference between bids and asks
    plt.fill_between(df.index, df['bidsUsd'], df['asksUsd'], alpha=0.3, 
                     where=(df['bidsUsd'] > df['asksUsd']), facecolor='green', interpolate=True)
    plt.fill_between(df.index, df['bidsUsd'], df['asksUsd'], alpha=0.3, 
                     where=(df['asksUsd'] > df['bidsUsd']), facecolor='red', interpolate=True)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("USD Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

    # Plot bid/ask ratio
    plt.figure(figsize=(12, 6))
    ratio = df['bidsUsd'] / df['asksUsd']
    plt.plot(df.index, ratio, label='Bid/Ask Ratio', color='blue')
    plt.axhline(y=1, color='r', linestyle='--', label='Equilibrium')
    plt.title(f"{title} - Bid/Ask Ratio")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}_ratio.png")
    plt.close()

    # Plot bid and ask amounts
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['bidsAmount'], label='Bid Amount', color='green')
    plt.plot(df.index, df['asksAmount'], label='Ask Amount', color='red')
    plt.title(f"{title} - Bid and Ask Amounts")
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}_amounts.png")
    plt.close()
    
def plot_spot_taker_buy_sell_volume(df: pd.DataFrame, title: str, filename: str):
    """
    Plot spot taker buy/sell volume history data with correct date handling.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot volumes
    ax1.plot(df.index, df['buy'], label='Buy Volume', color='green')
    ax1.plot(df.index, df['sell'], label='Sell Volume', color='red')
    ax1.set_title(f"{title} - Volumes")
    ax1.set_ylabel("Volume")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Calculate and plot ratio
    ratio = df['buy'] / df['sell']
    ax2.plot(df.index, ratio, label='Buy/Sell Ratio', color='blue')
    ax2.axhline(y=1, color='r', linestyle='--', label='Equilibrium')
    ax2.set_title(f"{title} - Buy/Sell Ratio")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Ratio")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation
    
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

    # Create a heatmap of buy/sell ratio
    plt.figure(figsize=(12, 6))
    plt.imshow([ratio.values], cmap='RdYlGn', aspect='auto', extent=[mdates.date2num(df.index[0]), mdates.date2num(df.index[-1]), 0, 1])
    plt.colorbar(label='Buy/Sell Ratio')
    plt.title(f"{title} - Buy/Sell Ratio Heatmap")
    plt.xlabel("Date")
    plt.yticks([])
    
    # Format x-axis for heatmap
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation
    
    plt.tight_layout()
    plt.savefig(f"images/{filename}_heatmap.png")
    plt.close()
    
def plot_spot_aggregated_taker_buy_sell_volume(df: pd.DataFrame, title: str, filename: str):
    """
    Plot spot aggregated taker buy/sell volume history data.

    Args:
        df: DataFrame containing spot aggregated taker buy/sell volume history data
        title: Title for the plot
        filename: Filename to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot volumes
    ax1.plot(df.index, df['buy'], label='aggregated Buy Volume', color='green')
    ax1.plot(df.index, df['sell'], label='aggregated Sell Volume', color='red')
    ax1.set_title(f"{title} - Volumes")
    ax1.set_ylabel("Volume")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Calculate and plot ratio
    ratio = df['buy'] / df['sell']
    ax2.plot(df.index, ratio, label='Buy/Sell Ratio', color='blue')
    ax2.axhline(y=1, color='r', linestyle='--', label='Equilibrium')
    ax2.set_title(f"{title} - Buy/Sell Ratio")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Ratio")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation
    
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png")
    plt.close()

    # Create a heatmap of buy/sell ratio
    plt.figure(figsize=(12, 6))
    plt.imshow([ratio.values], cmap='RdYlGn', aspect='auto', extent=[mdates.date2num(df.index[0]), mdates.date2num(df.index[-1]), 0, 1])
    plt.colorbar(label='Buy/Sell Ratio')
    plt.title(f"{title} - Buy/Sell Ratio Heatmap")
    plt.xlabel("Date")
    plt.yticks([])
    
    # Format x-axis for heatmap
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation
    
    plt.tight_layout()
    plt.savefig(f"images/{filename}_heatmap.png")
    plt.close()
    
    
