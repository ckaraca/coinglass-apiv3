from typing import Optional

import pandas as pd
import requests
from typing import Optional

from .exceptions import (
    CoinglassAPIError,
    CoinglassRequestError,
    NoDataReturnedError,
    RateLimitExceededError,
)
from .parameters import CoinglassParameterValidation


class CoinglassAPIv3(CoinglassParameterValidation):
    """ Unofficial Python client for Coinglass API """

    def __init__(self, coinglass_secret: str):
        """
        Args:
            coinglass_secret: key from Coinglass, get one at
            https://www.coinglass.com/pricing
        """

        super().__init__()

        self.__coinglass_secret = coinglass_secret
        self._base_url = "https://open-api-v3.coinglass.com/api/"
        self._session = requests.Session()

    def _get(self, endpoint: str, params: dict | None = None, api_type: str = "futures") -> dict:
        if params:
            self.validate_params(params)

        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.__coinglass_secret
        }
        url = f"{self._base_url}{api_type}/{endpoint}"
        return self._session.request(
            method='GET',
            url=url,
            params=params,
            headers=headers,
            timeout=30
        ).json()

    @staticmethod
    def _create_dataframe(
            data: list[dict],
            time_col: str | None = None,
            unit: str | None = "ms",
            cast_objects_to_numeric: bool = False
    ) -> pd.DataFrame:
        """
        Create pandas DataFrame from a list of dicts

        Args:
            data: list of dicts
            time_col: name of time column in dict
            unit: unit of time column, specify None to use auto-resolver (default: ms)
            cast_objects_to_numeric: cast all object columns to numeric (default: False)

        Returns:
            pandas DataFrame
        """
        df = pd.DataFrame(data)

        if time_col:
            if time_col == "time":
                # Handle edge case of time column being named "time"
                df.rename(columns={"time": "t"}, inplace=True)
                time_col = "t"

            df["time"] = pd.to_datetime(df[time_col], unit=unit)
            df.drop(columns=[time_col], inplace=True)
            df.set_index("time", inplace=True, drop=True)

            if "t" in df.columns:
                # Drop additional "t" column if it exists
                df.drop(columns=["t"], inplace=True)

        if cast_objects_to_numeric:
            cols = df.columns[df.dtypes.eq('object')]
            df[cols] = df[cols].apply(pd.to_numeric)

        return df

    @staticmethod
    def _create_multiindex_dataframe(
            data: list[dict],
            list_key: str
    ) -> pd.DataFrame:
        """
        Create MultiIndex pandas DataFrame from a list of nested dicts

        Args:
            data: list of nested dicts
            list_key: key in dict that contains list of dicts

        Returns:
            dict of pandas DataFrame
        """
        flattened_data = {}

        # Flatten nested dicts
        for symbol_data in data:
            flattened_dict = {}
            for outer_key, outer_value in symbol_data.items():
                if isinstance(outer_value, list):
                    for exchange in outer_value:
                        ex = exchange["exchangeName"]
                        for inner_key, value in exchange.items():
                            flattened_dict[(outer_key, ex, inner_key)] = value
                else:
                    flattened_dict[outer_key] = outer_value

            # Remove non-tuple keys
            remove_keys = []
            for key in list(flattened_dict.keys()):
                if not isinstance(key, tuple):
                    remove_keys.append(key)

            for k in remove_keys:
                flattened_dict.pop(k, None)

            df = pd.DataFrame.from_dict(flattened_dict, orient="index")
            df.index = pd.MultiIndex.from_tuples(df.index)

            flattened_data[symbol_data[list_key]] = df

        return pd.concat(flattened_data, axis=1)

    @staticmethod
    def _flatten_dictionary(data: dict) -> dict:
        flattened_dict = {}

        for outer_key, outer_value in data.items():
            if isinstance(outer_value, dict):
                for inner_key, inner_value in outer_value.items():
                    if isinstance(inner_value, list):
                        flattened_dict[(outer_key, inner_key)] = inner_value
                    else:
                        flattened_dict[inner_key] = inner_value
            else:
                flattened_dict[(outer_key, 0)] = outer_value

        return flattened_dict

    @staticmethod
    def _check_for_errors(response: dict) -> None:
        """ Check for errors in response """
        if not response:
            raise CoinglassAPIError(
                status=500,
                err="Empty response received from the API"
            )
        if "success" not in response:
            raise CoinglassAPIError(
                status=response.get("status", 500),
                err=response.get("msg", "Unknown error")
            )
        # Handle case where API response is unsuccessful
        if not response["success"]:
            code, msg = int(response.get("code", 40001)), response.get("msg", "Unknown error")
            match code:
                case 50001:
                    raise RateLimitExceededError()
                case _:
                    raise CoinglassRequestError(code=code, msg=msg)
        # Handle case where API returns no data
        if "data" not in response:
            raise NoDataReturnedError()
        
        
## Cem Started from here:
    """General Section"""
    def supported_coins(self) -> pd.DataFrame:
        """
        Fetch the list of supported coins and return as a DataFrame
        """
        response = self._get(endpoint="supported-coins")
        data = response["data"]
        return pd.DataFrame(data, columns=["supported_coins"])
    
    def supported_exchange_pairs(self) -> pd.DataFrame:
        """
        Fetch the list of supported exchanges and trading pairs and return as a DataFrame
        """
        response = self._get(endpoint="supported-exchange-pairs")
        pairs_data = []
        for exchange, pairs in response["data"].items():
            for pair in pairs:
                pairs_data.append({
                    "exchange": exchange,
                    "instrumentId": pair["instrumentId"],
                    "baseAsset": pair["baseAsset"],
                    "quoteAsset": pair["quoteAsset"]
                })
        return pd.DataFrame(pairs_data)
    """Open Interest Section"""
    def ohlc_history(self, exchange: str, symbol: str, interval: str, limit: int = 1000, startTime: Optional[int] = None, endTime: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLC history data.
        Args:
            exchange: Exchange name (e.g., Binance)
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Time interval (e.g., 1d)
            limit: Number of data points to return (default 1000, max 4500)
            startTime: Start time in seconds
            endTime: End time in seconds
        Returns:
            pandas DataFrame with OHLC data
        """
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": startTime,
            "endTime": endTime,
        }
        response = self._get(endpoint="openInterest/ohlc-history", params=params)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)

    def ohlc_aggregated_history(self, symbol: str, interval: str, limit: int = 1000, startTime: Optional[int] = None, endTime: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch aggregated OHLC history data.
        Args:
            symbol: Trading coin (e.g., BTC)
            interval: Time interval (e.g., 1d)
            limit: Number of data points to return (default 1000, max 4500)
            startTime: Start time in seconds
            endTime: End time in seconds
        Returns:
            pandas DataFrame with aggregated OHLC data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": startTime,
            "endTime": endTime,
        }
        response = self._get(endpoint="openInterest/ohlc-aggregated-history", params=params)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)
    
    def ohlc_aggregated_stablecoin_margin_history(self, exchanges: str, symbol: str, interval: str, limit: int = 1000, startTime: Optional[int] = None, endTime: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch aggregated stablecoin margin OHLC history data.
        Args:
            exchanges: Comma separated string of exchange names (e.g., Binance)
            symbol: Trading coin (e.g., BTC)
            interval: Time interval (e.g., 1d)
            limit: Number of data points to return (default 1000, max 4500)
            startTime: Start time in seconds
            endTime: End time in seconds
        Returns:
            pandas DataFrame with aggregated stablecoin margin OHLC data
        """
        params = {
            "exchanges": exchanges,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": startTime,
            "endTime": endTime,
        }
        response = self._get(endpoint="openInterest/ohlc-aggregated-stablecoin-margin-history", params=params)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)
    
    def ohlc_aggregated_coin_margin_history(self, exchanges: str, symbol: str, interval: str, limit: int = 1000, startTime: Optional[int] = None, endTime: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch aggregated coin margin OHLC history data.
        Args:
            exchanges: Comma separated string of exchange names (e.g., Binance)
            symbol: Trading coin (e.g., BTC)
            interval: Time interval (e.g., 1d)
            limit: Number of data points to return (default 1000, max 4500)
            startTime: Start time in seconds
            endTime: End time in seconds
        Returns:
            pandas DataFrame with aggregated coin margin OHLC data
        """
        params = {
            "exchanges": exchanges,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": startTime,
            "endTime": endTime,
        }
        response = self._get(endpoint="openInterest/ohlc-aggregated-coin-margin-history", params=params)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)
    
    def exchange_list(self, symbol: str) -> pd.DataFrame:
        """
        Fetch open interest data for a coin from exchanges.

        Args:
            symbol: Trading coin (e.g., BTC)

        Returns:
            pandas DataFrame with open interest data from exchanges
        """
        params = {
            "symbol": symbol
        }
        response = self._get(endpoint="openInterest/exchange-list", params=params)
        
        print("Raw API response:", response)
        print("Type of response:", type(response))
        
        if not isinstance(response, dict):
            raise CoinglassAPIError(status=500, err=f"Expected dict response, got {type(response)}")
        
        if 'data' not in response:
            raise CoinglassAPIError(status=500, err="No 'data' field in API response")
        
        data = response["data"]
        
        print("Data from API:", data)
        print("Type of data:", type(data))
        
        if not isinstance(data, list):
            raise CoinglassAPIError(status=500, err=f"Expected list, got {type(data)}")
        
        if len(data) == 0:
            print("Warning: API returned an empty list")
            return pd.DataFrame()  # Return an empty DataFrame
        
        print("First item in data:", data[0])
        print("Type of first item:", type(data[0]))
        
        # Try to create DataFrame without specifying dtypes
        try:
            df = pd.DataFrame(data)
            print("DataFrame created successfully")
            print("DataFrame info:")
            df.info()
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            print("Data causing the error:", data)
            raise
        
    def exchange_history_chart(
        self,
        symbol: str = "BTC",
        range: str = "12h",
        unit: str = "USD"
    ) -> pd.DataFrame:
        """
        Retrieve historical open interest data for a cryptocurrency from exchanges.

        Args:
            symbol (str): Trading coin (e.g., BTC). Defaults to "BTC".
            range (str): Time range for data. Options: "all", "1m", "15m", "4h", "12h". Defaults to "12h".
            unit (str): Unit for returned data. Options: "USD" or "COIN". Defaults to "USD".

        Returns:
            pd.DataFrame: DataFrame containing historical open interest data.
        """
        endpoint = "openInterest/exchange-history-chart"
        params = {
            "symbol": symbol,
            "range": range,
            "unit": unit
        }

        # Validate parameters
        self.validate_params(params)

        response = self._get(endpoint, params=params)
        self._check_for_errors(response)

        if "data" in response:
            data = response["data"]
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["timeList"], unit="ms"),
                "price": data["priceList"]
            })

            for exchange, values in data["dataMap"].items():
                df[exchange] = values

            df.set_index("timestamp", inplace=True)
            return df
        else:
            raise NoDataReturnedError()
##
##    
## Funding Rate Section
    def funding_rate_ohlc_history(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch funding rate OHLC history data.

        Args:
            exchange: Exchange name (e.g., Binance)
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Time interval (e.g., 1d)
            limit: Number of data points to return (default 1000, max 4500)
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            pandas DataFrame with funding rate OHLC data
        """
        endpoint = "fundingRate/ohlc-history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time,
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)

    def oi_weight_ohlc_history(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch open interest-weight OHLC history data.

        Args:
            symbol: Trading coin (e.g., BTC)
            interval: Time interval (e.g., 1d)
            limit: Number of data points to return (default 1000, max 4500)
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            pandas DataFrame with OI weight OHLC data
        """
        endpoint = "fundingRate/oi-weight-ohlc-history"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time,
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)
    
    def vol_weight_ohlc_history(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch volume-weight OHLC history data.

        Args:
            symbol: Trading coin (e.g., BTC)
            interval: Time interval (e.g., 1d)
            limit: Number of data points to return (default 1000, max 4500)
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            pandas DataFrame with volume weight OHLC data
        """
        endpoint = "fundingRate/vol-weight-ohlc-history"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time,
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)
    
    def funding_rate_exchange_list(self, symbol: str = "BTC") -> pd.DataFrame:
        """
        Fetch funding rate data from exchanges for a specific symbol.

        Args:
            symbol: Trading coin (e.g., BTC)

        Returns:
            pandas DataFrame with funding rate data from exchanges
        """
        endpoint = "fundingRate/exchange-list"
        params = {"symbol": symbol}
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]

        # Process the data
        result = []
        for item in data:
            symbol = item["symbol"]
            for margin_type in ["usdtOrUsdMarginList", "tokenMarginList"]:
                for exchange_data in item.get(margin_type, []):
                    result.append({
                        "symbol": symbol,
                        "margin_type": "USDT/USD" if margin_type == "usdtOrUsdMarginList" else "Token",
                        "exchange": exchange_data.get("exchange", "Unknown"),
                        "funding_rate": exchange_data.get("fundingRate", exchange_data.get("currentFundingRate", None)),
                        "next_funding_time": pd.to_datetime(exchange_data.get("nextFundingTime", None), unit="ms", errors='coerce')
                    })

        return pd.DataFrame(result)
##
##    
## LIQUIDATION SECTION
##
##
    def liquidation_history(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data for both long and short liquidations of a trading pair on the exchange.

        Args:
            exchange: Exchange name (e.g., Binance, OKX)
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Time interval (e.g., 1d, 1h, 4h)
            limit: Number of data points to return (default 1000, max 4500)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            pandas DataFrame with liquidation history data
        """
        endpoint = "liquidation/v2/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time,
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)
    
    def liquidation_aggregated_history(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch aggregated historical data for both long and short liquidations of a coin on the exchange.

        Args:
            symbol: Trading coin (e.g., BTC)
            interval: Time interval (e.g., 1d, 1h, 4h)
            limit: Number of data points to return (default 1000, max 4500)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            pandas DataFrame with aggregated liquidation history data
        """
        endpoint = "liquidation/v2/aggregated-history"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time,
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="t", unit="s", cast_objects_to_numeric=True)
    
    def liquidation_coin_list(self, ex: str = "Binance") -> pd.DataFrame:
        """
        Fetch liquidation data for all coins on the specified exchange.

        Args:
            ex: Exchange name (e.g., Binance, OKX). Defaults to Binance.

        Returns:
            pandas DataFrame with liquidation data for all coins
        """
        endpoint = "liquidation/coin-list"
        params = {"ex": ex}
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return pd.DataFrame(data)
    
    def liquidation_exchange_list(self, symbol: str = "BTC", range: str = "1h") -> pd.DataFrame:
        """
        Fetch liquidation data for coins across all exchanges.

        Args:
            symbol: Trading coin (e.g., BTC). Defaults to BTC.
            range: Time range for data. Options: "1h", "4h", "12h", "24h". Defaults to "1h".

        Returns:
            pandas DataFrame with liquidation data for all exchanges
        """
        endpoint = "liquidation/exchange-list"
        params = {"symbol": symbol, "range": range}
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return pd.DataFrame(data)
##
## https://open-api-v3.coinglass.com/api/futures/liquidation/order 
## https://open-api-v3.coinglass.com/api/futures/liquidation/aggregate-heatmap
## In Standard version, not implemented yet
## 
    def liquidation_aggregated_heatmap_model2(self, symbol: str = "BTC", range: str = "3d") -> dict:
        """
        Fetch aggregated liquidation levels data for heatmap visualization.

        Args:
            symbol: Trading coin (e.g., BTC). Defaults to BTC.
            range: Time range for data. Options: "12h", "24h", "3d", "7d", "30d", "90d", "180d", "1y". Defaults to "3d".

        Returns:
            dict: Contains 'y' (price list), 'liq' (liquidation data), and 'prices' (OHLC data)
        """
        endpoint = "liquidation/model2/aggregate-heatmap"
        params = {"symbol": symbol, "range": range}
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        return response["data"]

    def liquidation_heatmap_model2(self, exchange: str = "Binance", symbol: str = "BTCUSDT", range: str = "3d") -> dict:
        """
        Fetch liquidation levels data for heatmap visualization.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            range: Time range for data. Options: "12h", "24h", "3d", "7d", "30d", "90d", "180d", "1y". Defaults to "3d".

        Returns:
            dict: Contains 'price_levels' (DataFrame), 'liquidations' (DataFrame), and 'ohlc' (DataFrame)
        """
        endpoint = "liquidation/model2/heatmap"
        params = {"exchange": exchange, "symbol": symbol, "range": range}
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]

        # Create DataFrame for price levels
        price_levels = pd.DataFrame({'price': data['y']})

        # Create DataFrame for liquidation data
        liquidations = pd.DataFrame(data['liq'], columns=['time_index', 'price_index', 'liquidation_value'])
        liquidations['price'] = liquidations['price_index'].map(lambda x: data['y'][x])

        # Create DataFrame for OHLC data
        ohlc = pd.DataFrame(data['prices'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ohlc['timestamp'] = pd.to_datetime(ohlc['timestamp'], unit='s')
        ohlc.set_index('timestamp', inplace=True)
        
        # Convert time_index to datetime
        liquidations['time'] = pd.to_datetime(liquidations['time_index'], unit='s')
        
        # Merge liquidations with OHLC data
        power_bi_data = pd.merge_asof(liquidations.sort_values('time'), ohlc.reset_index(), left_on='time', right_on='timestamp', direction='nearest')
        
        # Add exchange and symbol columns
        power_bi_data['exchange'] = exchange
        power_bi_data['symbol'] = symbol
        
        # Export data to CSV for Power BI
        #csv_filename = f"images/{exchange}_{symbol}_liquidation_heatmap_{range}.csv"
        #power_bi_data.to_csv(csv_filename, index=False)
        #print(f"Data exported to {csv_filename} for Power BI")

        return {
            'price_levels': price_levels,
            'liquidations': liquidations,
            'ohlc': ohlc
        }
        
    

##    
## Long Short Account Ratio Section
##      
    
    def global_long_short_account_ratio(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "h1",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve the global long/short account ratio for trading pairs on an exchange.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with long/short account ratio data
        """
        endpoint = "globalLongShortAccountRatio/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def top_long_short_account_ratio(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "h1",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve the long/short ratio history for top accounts on an exchange.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with top accounts long/short ratio data
        """
        endpoint = "topLongShortAccountRatio/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def top_long_short_position_ratio_history(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "h1",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data for the long/short ratio of positions by top accounts.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with top accounts long/short ratio data.
        """
        endpoint = "topLongShortPositionRatio/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def aggregated_taker_buy_sell_volume_ratio_history(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data for the long/short ratio of aggregated taker buy/sell volumes.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with long/short ratio data for aggregated taker buy/sell volumes
        """
        endpoint = "aggregatedTakerBuySellVolumeRatio/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def aggregated_taker_buy_sell_volume_history(
        self,
        exchanges: str = "Binance",
        symbol: str = "BTC",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data for the aggregated taker buy/sell volumes.

        Args:
            exchanges: Comma-separated list of exchange names (e.g., "Binance,OKX,Bybit"). Defaults to "Binance".
            symbol: Trading coin (e.g., BTC). Defaults to BTC.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with buy and sell volume data
        """
        endpoint = "aggregatedTakerBuySellVolume/history"
        params = {
            "exchanges": exchanges,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def taker_buy_sell_volume_history(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data for the taker buy/sell volumes for a specific exchange and trading pair.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with buy and sell volume data
        """
        endpoint = "takerBuySellVolume/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def exchange_taker_buy_sell_ratio(self, symbol: str = "BTC", range: str = "1h") -> dict:
        """
        Retrieve the long/short ratio of aggregated taker buy/sell volumes for exchanges.

        Args:
            symbol: Trading coin (e.g., BTC). Defaults to BTC.
            range: Time range for data. Options: "5m", "15m", "30m", "1h", "4h", "12h", "24h". Defaults to "1h".

        Returns:
            dict: Contains overall data and list of exchange-specific data
        """
        endpoint = "takerBuySellVolume/exchange-list"
        params = {"symbol": symbol, "range": range}
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        return response["data"]
## GLOBAL SECTION
##
##
    # Requires Standard API Key
    def coins_markets(self, exchanges: str = "Binance,OKX", page_num: int = 1, page_size: int = 100) -> pd.DataFrame:
        """
        Retrieve performance-related information for all available coins.

        Args:
            exchanges: Comma-separated list of exchange names. Defaults to "Binance,OKX".
            page_num: Page number for pagination. Defaults to 1.
            page_size: Number of items per page. Defaults to 100.

        Returns:
            pandas DataFrame containing coin market data
        """
        endpoint = "coins-markets"
        params = {
            "exchanges": exchanges,
            "pageNum": page_num,
            "pageSize": page_size
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return pd.DataFrame(data)
    
    def pairs_markets(self, symbol: str = "BTC") -> pd.DataFrame:
        """
        Retrieve performance-related information for all available trading pairs of a specific coin.

        Args:
            symbol: Symbol of the coin (e.g., "BTC"). Defaults to "BTC".

        Returns:
            pandas DataFrame containing pairs market data
        """
        endpoint = "pairs-markets"
        params = {"symbol": symbol}
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        
        # Extract the nested data
        nested_data = response.get("data", {})
        if isinstance(nested_data, dict) and "data" in nested_data:
            data = nested_data["data"]
        else:
            raise CoinglassAPIError(status=500, err="Unexpected data structure in API response")
        
        if not data:
            raise NoDataReturnedError()
        
        return pd.DataFrame(data)
    
    def orderbook_history(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data of the order book for futures trading.

        Args:
            exchange: Exchange name (e.g., Binance, OKX)
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Time interval (e.g., 1h, 4h, 1d)
            limit: Number of data points to return (default 1000, max 4500)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            pandas DataFrame with orderbook history data
        """
        endpoint = "orderbook/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def aggregated_orderbook_history(
        self,
        exchanges: str = "Binance",
        symbol: str = "BTC",
        interval: str = "1h",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data of the aggregated order book for futures trading.

        Args:
            exchanges: Comma-separated list of exchange names (e.g., "Binance,OKX,Bybit")
            symbol: Trading coin (e.g., BTC)
            interval: Time interval (e.g., 1h, 4h, 1d)
            limit: Number of data points to return (default 1000, max 4500)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            pandas DataFrame with aggregated orderbook history data
        """
        endpoint = "orderbook/aggregated-history"
        params = {
            "exchanges": exchanges,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params)
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
## SPOT SECTION
##
    def spot_supported_coins(self) -> list:
        """
        Retrieve the list of supported coins for spot trading.

        Returns:
            list: A list of supported coin symbols
        """
        endpoint = "supported-coins"
        response = self._get(endpoint, api_type="spot")
        self._check_for_errors(response)
        return response["data"]
    
    def spot_supported_exchange_pairs(self) -> dict:
        """
        Retrieve the supported exchanges and their trading pairs for spot trading.

        Returns:
            dict: A dictionary where keys are exchange names and values are lists of dictionaries
                  containing information about supported trading pairs.
        """
        endpoint = "supported-exchange-pairs"
        response = self._get(endpoint, api_type="spot")
        self._check_for_errors(response)
        return response["data"]
    
    def spot_taker_buy_sell_volume_history(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data for the long/short ratio of taker buy/sell volumes in spot markets.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with taker buy/sell volume data for spot markets
        """
        endpoint = "takerBuySellVolume/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params, api_type="spot")
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def spot_aggregated_taker_buy_sell_volume_history(
        self,
        exchanges: str = "Binance",
        symbol: str = "BTC",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data for the aggregated taker buy/sell volumes in spot markets.

        Args:
            exchanges: Comma-separated list of exchange names (e.g., "Binance,OKX,Bybit"). Defaults to "Binance".
            symbol: Trading coin (e.g., BTC). Defaults to BTC.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with aggregated taker buy/sell volume data for spot markets
        """
        endpoint = "aggregatedTakerBuySellVolume/history"
        params = {
            "exchanges": exchanges,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params, api_type="spot")
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def spot_orderbook_history(
        self,
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data of the order book for spot trading.

        Args:
            exchange: Exchange name (e.g., Binance, OKX). Defaults to Binance.
            symbol: Trading pair (e.g., BTCUSDT). Defaults to BTCUSDT.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1d.
            limit: Number of data points to return. Default 1000, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with spot orderbook history data
        """
        endpoint = "orderbook/history"
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params, api_type="spot")
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def spot_aggregated_orderbook_history(
        self,
        exchanges: str = "Binance,OKX,Bybit",
        symbol: str = "BTC",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data of the aggregated order book for spot trading.

        Args:
            exchanges: Comma-separated list of exchange names. Defaults to "Binance,OKX,Bybit".
            symbol: Trading coin (e.g., BTC). Defaults to BTC.
            interval: Time interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w. Defaults to 1h.
            limit: Number of data points to return. Default 500, Max 4500.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).

        Returns:
            pandas DataFrame with aggregated spot orderbook history data
        """
        endpoint = "orderbook/aggregated-history"
        params = {
            "exchanges": exchanges,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = self._get(endpoint, params=params, api_type="spot")
        self._check_for_errors(response)
        data = response["data"]
        return self._create_dataframe(data, time_col="time", unit="s", cast_objects_to_numeric=True)
    
    def spot_pairs_markets(self, symbol: str = "BTC") -> pd.DataFrame:
        """
        Retrieve performance-related information for all available spot trading pairs of a specific coin.

        Args:
            symbol: Symbol of the coin (e.g., "BTC"). Defaults to "BTC".

        Returns:
            pandas DataFrame containing spot pairs market data
        """
        endpoint = "pairs-markets"
        params = {"symbol": symbol}
        response = self._get(endpoint, params=params, api_type="spot")
        self._check_for_errors(response)
        data = response["data"]
        return pd.DataFrame(data)
##
## INDICATOR SECTION
##
    def bitcoin_bubble_index(self) -> pd.DataFrame:
        """
        Fetch the Bitcoin Bubble Index data.

        Returns:
            pandas DataFrame with Bitcoin Bubble Index data
        """
        endpoint = "bitcoin-bubble-index"
        #response = self._get(endpoint)
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    
    def ahr999_index(self) -> pd.DataFrame:
        """
        Fetch the AHR999 index data.

        Returns:
            pandas DataFrame with AHR999 index data
        """
        endpoint = "ahr999"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    
    def two_year_ma_multiplier(self) -> pd.DataFrame:
        """
        Fetch the Two Year MA Multiplier data.

        Returns:
            pandas DataFrame with Two Year MA Multiplier data
        """
        endpoint = "tow-year-ma-multiplier"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['createTime'] = pd.to_datetime(df['createTime'], unit='ms')
        df.set_index('createTime', inplace=True)
        return df
    
    def two_hundred_week_moving_avg_heatmap(self) -> pd.DataFrame:
        """
        Fetch the 200-Week Moving Average Heatmap data.
        """
        endpoint = "tow-hundred-week-moving-avg-heatmap"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['createTime'] = pd.to_datetime(df['createTime'], unit='ms')
        df.set_index('createTime', inplace=True)
        return df

    def puell_multiple(self) -> pd.DataFrame:
        """
        Fetch the Puell Multiple data.
        """
        endpoint = "puell-multiple"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['createTime'] = pd.to_datetime(df['createTime'], unit='ms')
        df.set_index('createTime', inplace=True)
        return df

    def stock_to_flow(self) -> pd.DataFrame:
        """
        Fetch the Stock-to-Flow Model data.
        """
        endpoint = "stock-flow"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['createTime'] = pd.to_datetime(df['createTime'])
        df.set_index('createTime', inplace=True)
        return df

    def pi_cycle_top_indicator(self) -> pd.DataFrame:
        """
        Fetch the Pi Cycle Top Indicator data.
        """
        endpoint = "pi"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['createTime'] = pd.to_datetime(df['createTime'], unit='ms')
        df.set_index('createTime', inplace=True)
        return df
    
    def golden_ratio_multiplier(self) -> pd.DataFrame:
        """
        Fetch the Golden Ratio Multiplier data.
        """
        endpoint = "golden-ratio-multiplier"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['createTime'] = pd.to_datetime(df['createTime'], unit='ms')
        df.set_index('createTime', inplace=True)
        return df
    
    def bitcoin_profitable_days(self) -> pd.DataFrame:
        """
        Fetch the Bitcoin Profitable Days data.
        """
        endpoint = "bitcoin-profitable-days"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data)
        df['createTime'] = pd.to_datetime(df['createTime'], unit='ms')
        df.set_index('createTime', inplace=True)
        return df
    
    def bitcoin_rainbow_chart(self) -> pd.DataFrame:
        """
        Fetch the Bitcoin Rainbow Chart data.
        """
        endpoint = "bitcoin-rainbow-chart"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        df = pd.DataFrame(data, columns=['price', 'model_price', 'fire_sale', 'buy', 'accumulate', 'still_cheap', 'hold', 'is_this_a_bubble', 'fomo_intensifies', 'sell_seriously_sell', 'maximum_bubble_territory', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def fear_greed_index(self) -> pd.DataFrame:
        """
        Fetch the Crypto Fear & Greed Index data.
        """
        endpoint = "fear-greed-history"
        response = self._get(endpoint, api_type="index")
        self._check_for_errors(response)
        data = response["data"]
        
        # Check if the expected keys are in the data
        if not all(key in data for key in ['values', 'dates']):
            raise ValueError("Unexpected data structure in Fear & Greed Index response")
        
        df = pd.DataFrame({
            'values': data['values'],
            'dates': pd.to_datetime(data['dates'], unit='ms')  # Changed 's' to 'ms'
        })
        df.set_index('dates', inplace=True)
        
        # If 'prices' data is available, include it (note the plural 'prices')
        if 'prices' in data:
            df['price'] = data['prices']
        
        return df