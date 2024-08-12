# Coinglass API v3

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## Unofficial Python client for Coinglass API v3

This project is a fork of the original [Coinglass API wrapper](https://github.com/dineshpinto/coinglass-api) by Dinesh Pinto, updated to support Coinglass API v3.

This wrapper fetches data about crypto derivatives from the [Coinglass API v3](https://coinglass.com/pricing). All data is output in pandas DataFrames (single or multi-index) and all time-series data uses a `DateTimeIndex`. It supports all Coinglass API v3 endpoints.

## Installation

Currently, this project is not available via pip. To use it, clone this repository:

```bash
git clone https://github.com/your-username/coinglass-api-v3.git
cd coinglass-api-v3

## Usage

from coinglass_api import CoinglassAPIv3

cg = CoinglassAPIv3(api_key="your_api_key_here")

# Get supported coins
supported_coins = cg.supported_coins()

# Get supported exchanges and pairs
supported_pairs = cg.supported_exchange_pairs()

# Get OHLC history for BTC/USDT on Binance
ohlc_history = cg.ohlc_history(exchange="Binance", symbol="BTCUSDT", interval="1d", limit=10)

# Get aggregated OHLC history for BTC
ohlc_agg_history = cg.ohlc_aggregated_history(symbol="BTC", interval="1d", limit=10)

# Get aggregated stablecoin margin OHLC history for BTC
ohlc_agg_stablecoin = cg.ohlc_aggregated_stablecoin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=10)

# Get aggregated coin margin OHLC history for BTC
ohlc_agg_coin = cg.ohlc_aggregated_coin_margin_history(exchanges="Binance", symbol="BTC", interval="1d", limit=10)

# Get open interest data from exchanges for BTC
exchange_list = cg.exchange_list(symbol="BTC")

# Get exchange history chart for BTC
exchange_history = cg.exchange_history_chart(symbol="BTC", range="4h", unit="USD")

# and more...

## Examples
>>> cg.ohlc_history(exchange="Binance", symbol="BTCUSDT", interval="1d", limit=5).head()

Disclaimer
This project is for educational purposes only. You should not construe any such information or other material as legal, tax, investment, financial, or other advice. Nothing contained here constitutes a solicitation, recommendation, endorsement, or offer by me or any third party service provider to buy or sell any securities or other financial instruments in this or in any other jurisdiction in which such solicitation or offer would be unlawful under the securities laws of such jurisdiction.
Under no circumstances will I be held responsible or liable in any way for any claims, damages, losses, expenses, costs, or liabilities whatsoever, including, without limitation, any direct or indirect damages for loss of profits.
