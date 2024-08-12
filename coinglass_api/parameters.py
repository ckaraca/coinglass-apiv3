import warnings

from .exceptions import CoinglassParameterWarning


class CoinglassParameterValidation:
    def __init__(self):
        self._exchanges: list[str] = [
            'Binance', 'OKX', 'dYdX', 'Bitget', 'Bybit', 'BingX', 'Bitmex', 'Bitfinex',
            'Deribit', 'CoinEx', 'Kraken', 'Huobi'
        ]

        self._time_types: list[str] = [
            'h1', 'h2', 'h4', 'h6', 'h8', 'h12', 'h24', 'm1', 'm3', 'm5', 'm15', 'm30',
            '1d', '7d'
        ]
        
        self._ranges = ["all", "1m", "15m", "4h", "12h"]
        self._units = ["USD", "COIN"]

    def add_exchange(self, exchange: str):
        self._exchanges.append(exchange)

    def add_time_type(self, time_type: str):
        self._time_types.append(time_type)

    def get_exchanges(self) -> list[str]:
        """ Returns list of exchanges """
        return self._exchanges

    def get_time_types(self) -> list[str]:
        """ Returns list of time types """
        return self._time_types

    def _validate_exchange(self, exchange: str):
        if exchange not in self._exchanges:
            warnings.warn(
                f"'{exchange}' not in exchange list: {self._exchanges}",
                CoinglassParameterWarning,
                stacklevel=2
            )

    def _validate_time_type(self, time_type: str):
        if time_type not in self._time_types:
            warnings.warn(
                f"'{time_type}' not in time type list: {self._time_types}",
                CoinglassParameterWarning,
                stacklevel=2
            )
            
    def _validate_range(self, range: str):
        if range not in self._ranges:
            warnings.warn(
                f"'{range}' not in range list: {self._ranges}",
                CoinglassParameterWarning,
                stacklevel=2
            )

    def _validate_unit(self, unit: str):
        if unit not in self._units:
            warnings.warn(
                f"'{unit}' not in unit list: {self._units}",
                CoinglassParameterWarning,
                stacklevel=2
            )


    def validate_params(self, params: dict):
        if "ex" in params:
            self._validate_exchange(params["ex"])
        elif "ex_name" in params:
            self._validate_exchange(params["ex_name"])
        if "time_type" in params:
            self._validate_time_type(params["time_type"])
        elif "interval" in params:
            self._validate_time_type(params["interval"])
        if "range" in params:
            self._validate_range(params["range"])
        if "unit" in params:
            self._validate_unit(params["unit"])