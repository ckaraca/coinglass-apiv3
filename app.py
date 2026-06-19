
from flask import Flask, jsonify, request
from coinglass_api.api import CoinglassAPIv3
import os
import pandas as pd
import json

app = Flask(__name__)

# --- Helper Function ---
def dataframe_to_json(df):
    """Converts a pandas DataFrame or a dictionary of DataFrames to a JSON serializable format."""
    if isinstance(df, pd.DataFrame):
        return json.loads(df.to_json(orient='records', date_format='iso'))
    if isinstance(df, dict):
        for key, value in df.items():
            if isinstance(value, pd.DataFrame):
                df[key] = json.loads(value.to_json(orient='records', date_format='iso'))
        return df
    return df

# --- API Initialization ---
try:
    api_key = os.environ.get('COINGLASS_API_KEY')
    if not api_key:
        raise ValueError("COINGLASS_API_KEY environment variable not set.")
    cg_api = CoinglassAPIv3(coinglass_secret=api_key)
except (ValueError, Exception) as e:
    print(f"Error during API initialization: {e}")
    cg_api = None

# --- API Endpoints ---
@app.route('/')
def index():
    """Provides a list of available endpoints."""
    return jsonify({
        "message": "Welcome to the Coinglass API Wrapper!",
        "endpoints": [
            "/",
            "/liquidation/history",
            "/liquidation/aggregated-history",
            "/liquidation/coin-list",
            "/liquidation/exchange-list",
            "/long-short-ratio/global-history",
            "/long-short-ratio/top-account-history",
            "/long-short-ratio/top-position-history",
            "/funding-rate/ohlc-history",
            "/open-interest/ohlc-history",
            "/global/coins-markets",
            "/global/pairs-markets",
            "/spot/pairs-markets",
            "/indicator/bitcoin-bubble-index",
            "/indicator/fear-greed-index",
            "/indicator/ahr999-index",
            "/indicator/two-year-ma-multiplier",
            "/indicator/puell-multiple"
        ]
    })

# --- Liquidation Endpoints ---
@app.route('/liquidation/history')
def liquidation_history():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "exchange": request.args.get('exchange', default='Binance', type=str),
            "symbol": request.args.get('symbol', default='BTCUSDT', type=str),
            "interval": request.args.get('interval', default='h1', type=str),
            "limit": request.args.get('limit', default=100, type=int)
        }
        data = cg_api.liquidation_history(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/liquidation/aggregated-history')
def liquidation_aggregated_history():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "symbol": request.args.get('symbol', default='BTC', type=str),
            "interval": request.args.get('interval', default='h1', type=str),
            "limit": request.args.get('limit', default=100, type=int)
        }
        data = cg_api.liquidation_aggregated_history(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/liquidation/coin-list')
def liquidation_coin_list():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        exchange = request.args.get('ex', default='Binance', type=str)
        data = cg_api.liquidation_coin_list(ex=exchange)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/liquidation/exchange-list')
def liquidation_exchange_list():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        symbol = request.args.get('symbol', default='BTC', type=str)
        time_range = request.args.get('range', default='1h', type=str)
        data = cg_api.liquidation_exchange_list(symbol=symbol, range=time_range)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Long/Short Ratio Endpoints ---
@app.route('/long-short-ratio/global-history')
def global_long_short_account_ratio():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "exchange": request.args.get('exchange', default='Binance', type=str),
            "symbol": request.args.get('symbol', default='BTCUSDT', type=str),
            "interval": request.args.get('interval', default='h1', type=str),
            "limit": request.args.get('limit', default=100, type=int)
        }
        data = cg_api.global_long_short_account_ratio(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/long-short-ratio/top-account-history')
def top_long_short_account_ratio():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "exchange": request.args.get('exchange', default='Binance', type=str),
            "symbol": request.args.get('symbol', default='BTCUSDT', type=str),
            "interval": request.args.get('interval', default='h1', type=str),
            "limit": request.args.get('limit', default=100, type=int)
        }
        data = cg_api.top_long_short_account_ratio(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/long-short-ratio/top-position-history')
def top_long_short_position_ratio_history():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "exchange": request.args.get('exchange', default='Binance', type=str),
            "symbol": request.args.get('symbol', default='BTCUSDT', type=str),
            "interval": request.args.get('interval', default='h1', type=str),
            "limit": request.args.get('limit', default=100, type=int)
        }
        data = cg_api.top_long_short_position_ratio_history(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Funding/Open Interest Endpoints ---
@app.route('/funding-rate/ohlc-history')
def funding_rate_ohlc_history():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "exchange": request.args.get('exchange', default='Binance', type=str),
            "symbol": request.args.get('symbol', default='BTCUSDT', type=str),
            "interval": request.args.get('interval', default='h1', type=str),
            "limit": request.args.get('limit', default=100, type=int)
        }
        data = cg_api.funding_rate_ohlc_history(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/open-interest/ohlc-history')
def open_interest_ohlc_history():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "exchange": request.args.get('exchange', default='Binance', type=str),
            "symbol": request.args.get('symbol', default='BTCUSDT', type=str),
            "interval": request.args.get('interval', default='h1', type=str),
            "limit": request.args.get('limit', default=100, type=int)
        }
        data = cg_api.ohlc_history(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Global and Spot Market Endpoints ---
@app.route('/global/coins-markets')
def coins_markets():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        params = {
            "exchanges": request.args.get('exchanges', default='Binance,OKX', type=str),
            "page_num": request.args.get('page_num', default=1, type=int),
            "page_size": request.args.get('page_size', default=100, type=int)
        }
        data = cg_api.coins_markets(**params)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/global/pairs-markets')
def pairs_markets():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        symbol = request.args.get('symbol', default='BTC', type=str)
        data = cg_api.pairs_markets(symbol=symbol)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/spot/pairs-markets')
def spot_pairs_markets():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        symbol = request.args.get('symbol', default='BTC', type=str)
        data = cg_api.spot_pairs_markets(symbol=symbol)
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Indicator Endpoints ---
@app.route('/indicator/bitcoin-bubble-index')
def bitcoin_bubble_index():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        data = cg_api.bitcoin_bubble_index()
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/indicator/fear-greed-index')
def fear_greed_index():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        data = cg_api.fear_greed_index()
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/indicator/ahr999-index')
def ahr999_index():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        data = cg_api.ahr999_index()
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/indicator/two-year-ma-multiplier')
def two_year_ma_multiplier():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        data = cg_api.two_year_ma_multiplier()
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/indicator/puell-multiple')
def puell_multiple():
    if not cg_api: return jsonify({"error": "API not initialized"}), 500
    try:
        data = cg_api.puell_multiple()
        return jsonify(dataframe_to_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Note: This is for local development and testing.
    # Vercel will use a production WSGI server.
    app.run(host='0.0.0.0', port=8080)
