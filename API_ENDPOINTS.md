# Coinglass Data API Endpoints

This document describes the API endpoints available from the Coinglass Data API service.

## Endpoints

### GET /liquidation_data/<exchange>/<symbol>/<range>/<interval>/<limit>

**Description:** Fetches liquidation heatmap data for a given exchange, symbol, range, interval, and limit. This data is useful for identifying significant price levels where large liquidations occurred.

**Parameters:**

*   `exchange` (string, **required**): The exchange name (e.g., "Binance", "OKX", "Bybit"). Case-sensitive, should match the exchange names supported by the Coinglass API.
*   `symbol` (string, **required**): The trading pair symbol (e.g., "BTCUSDT", "ETHUSDT"). Case-sensitive, should match the symbols supported by the exchange and Coinglass API.
*   `range` (string, **required**): The time range for the data. Possible values depend on the Coinglass API, commonly include: "12h", "24h", "3d", "7d", "30d", "90d", "180d", "1y".
*   `interval` (string, **required**): The time interval for the data points within the specified range. Possible values depend on the Coinglass API and range, commonly include: "1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d".
*   `limit` (integer, **required**): The maximum number of data points to return. Should be a positive integer.

**Example Request:**

```
GET /liquidation_data/Binance/BTCUSDT/24h/1h/100
```
**Response (JSON):**

A list of dictionaries, where each dictionary represents a liquidation data point. The exact keys in the dictionary will depend on the data returned by the Coinglass API's `liquidation_heatmap_model2` function, but typically include:
```
json
[
  {
    "time_index": 1678886400,
    "price_index": 100,
    "liquidation_value": 1500000.50,
    "price": 27500.25,
    "time": "2023-03-15T12:00:00Z",
    "timestamp": "2023-03-15T12:00:00Z",
    "open": 27400.00,
    "high": 27600.00,
    "low": 27350.00,
    "close": 27500.00,
    "volume": 1000.50,
    "exchange": "Binance",
    "symbol": "BTCUSDT"
  },
  {
    "time_index": 1678890000,
    "price_index": 110,
    "liquidation_value": 2000000.00,
    "price": 28000.00,
     "time": "2023-03-15T13:00:00Z",
    "timestamp": "2023-03-15T13:00:00Z",
    "open": 27500.00,
    "high": 28100.00,
    "low": 27450.00,
    "close": 28000.00,
    "volume": 1500.75,
    "exchange": "Binance",
    "symbol": "BTCUSDT"
  }
  // ... more data points
]
```
*(Note: The exact keys and the format of time-related fields might vary slightly based on the `coinglass_api` library's processing.)*

**Error Responses:**

*   `400 Bad Request`: Returned if the `limit` parameter is not a valid integer.
```
json
{
  "error": "Invalid limit parameter. Must be an integer."
}
```
*   `404 Not Found`: Returned if no liquidation data is available for the combination of provided parameters.
```
json
{
  "message": "No liquidation data available for the given parameters"
}
```
*   `500 Internal Server Error`: Returned if there is an issue with the Coinglass API or an unexpected server error occurs during processing.
```
json
{
  "error": "Details about the API error or server issue."
}
```
## Future Endpoints (Under Development)

*   `/ohlc_data/<exchange>/<symbol>/<interval>/<limit>`: Historical OHLC data.
*   `/funding_rates/<symbol>`: Current funding rates across exchanges.
*   `/open_interest/<symbol>`: Open interest data across exchanges.
*   `/long_short_ratio/<exchange>/<symbol>/<interval>/<limit>`: Long/Short ratio data.