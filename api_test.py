
import os
import unittest
from unittest.mock import patch, MagicMock
import json

# Set a dummy API key for testing purposes
os.environ['COINGLASS_API_KEY'] = 'DUMMY_API_KEY'

from app import app, dataframe_to_json
from coinglass_api.api import CoinglassAPIv3
import pandas as pd

class APITestCase(unittest.TestCase):
    """Test suite for the Flask API endpoints."""

    def setUp(self):
        """Set up test client and mock the Coinglass API."""
        self.app = app.test_client()
        self.app.testing = True

        # Mock the entire CoinglassAPIv3 class
        self.mock_cg_api = patch('app.cg_api').start()
        self.addCleanup(patch.stopall)

    def test_01_index_route(self):
        """Test the index route returns the list of endpoints."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('endpoints', data)
        self.assertIsInstance(data['endpoints'], list)

    def _test_endpoint(self, endpoint, mock_function_name, expected_params, query_params=''):
        """Helper function to test a generic endpoint."""
        mock_df = pd.DataFrame({'test_col': [1, 2], 'test_val': ['A', 'B']})
        getattr(self.mock_cg_api, mock_function_name).return_value = mock_df

        response = self.app.get(f'{endpoint}{query_params}')
        self.assertEqual(response.status_code, 200)
        
        getattr(self.mock_cg_api, mock_function_name).assert_called_with(**expected_params)
        
        # Use the app's dataframe_to_json for consistent testing
        self.assertEqual(response.get_json(), dataframe_to_json(mock_df))

    def _test_indicator_endpoint(self, endpoint, mock_function_name):
        """Helper function to test an indicator endpoint."""
        mock_df = pd.DataFrame({'value': [100]})
        getattr(self.mock_cg_api, mock_function_name).return_value = mock_df
        response = self.app.get(endpoint)
        self.assertEqual(response.status_code, 200)
        getattr(self.mock_cg_api, mock_function_name).assert_called_once()
        self.assertEqual(response.get_json(), dataframe_to_json(mock_df))

    def test_02_liquidation_history(self):
        self._test_endpoint(
            '/liquidation/history',
            'liquidation_history',
            {'exchange': 'Binance', 'symbol': 'BTCUSDT', 'interval': 'h1', 'limit': 100}
        )

    def test_03_liquidation_aggregated_history(self):
        self._test_endpoint(
            '/liquidation/aggregated-history',
            'liquidation_aggregated_history',
            {'symbol': 'BTC', 'interval': 'h1', 'limit': 100}
        )

    def test_04_liquidation_coin_list(self):
        self._test_endpoint(
            '/liquidation/coin-list',
            'liquidation_coin_list',
            {'ex': 'Binance'}
        )

    def test_05_liquidation_exchange_list(self):
        self._test_endpoint(
            '/liquidation/exchange-list',
            'liquidation_exchange_list',
            {'symbol': 'BTC', 'range': '1h'}
        )

    def test_06_global_long_short_ratio(self):
        self._test_endpoint(
            '/long-short-ratio/global-history',
            'global_long_short_account_ratio',
            {'exchange': 'Binance', 'symbol': 'BTCUSDT', 'interval': 'h1', 'limit': 100}
        )

    def test_07_top_account_long_short_ratio(self):
        self._test_endpoint(
            '/long-short-ratio/top-account-history',
            'top_long_short_account_ratio',
            {'exchange': 'Binance', 'symbol': 'BTCUSDT', 'interval': 'h1', 'limit': 100}
        )

    def test_08_top_position_long_short_ratio(self):
        self._test_endpoint(
            '/long-short-ratio/top-position-history',
            'top_long_short_position_ratio_history',
            {'exchange': 'Binance', 'symbol': 'BTCUSDT', 'interval': 'h1', 'limit': 100}
        )

    def test_09_funding_rate_history(self):
        self._test_endpoint(
            '/funding-rate/ohlc-history',
            'funding_rate_ohlc_history',
            {'exchange': 'Binance', 'symbol': 'BTCUSDT', 'interval': 'h1', 'limit': 100}
        )

    def test_10_open_interest_history(self):
        self._test_endpoint(
            '/open-interest/ohlc-history',
            'ohlc_history',
            {'exchange': 'Binance', 'symbol': 'BTCUSDT', 'interval': 'h1', 'limit': 100}
        )

    def test_11_coins_markets(self):
        self._test_endpoint(
            '/global/coins-markets',
            'coins_markets',
            {'exchanges': 'Binance,OKX', 'page_num': 1, 'page_size': 100},
            query_params='?exchanges=Binance,OKX&page_num=1&page_size=100'
        )

    def test_12_pairs_markets(self):
        self._test_endpoint(
            '/global/pairs-markets',
            'pairs_markets',
            {'symbol': 'ETH'},
            query_params='?symbol=ETH'
        )

    def test_13_spot_pairs_markets(self):
        self._test_endpoint(
            '/spot/pairs-markets',
            'spot_pairs_markets',
            {'symbol': 'BTC'}
        )

    def test_14_bitcoin_bubble_index(self):
        self._test_indicator_endpoint('/indicator/bitcoin-bubble-index', 'bitcoin_bubble_index')

    def test_15_fear_greed_index(self):
        self._test_indicator_endpoint('/indicator/fear-greed-index', 'fear_greed_index')

    def test_16_ahr999_index(self):
        self._test_indicator_endpoint('/indicator/ahr999-index', 'ahr999_index')

    def test_17_two_year_ma_multiplier(self):
        self._test_indicator_endpoint('/indicator/two-year-ma-multiplier', 'two_year_ma_multiplier')

    def test_18_puell_multiple(self):
        self._test_indicator_endpoint('/indicator/puell-multiple', 'puell_multiple')

if __name__ == "__main__":
    # Note: Setting the environment variable is crucial for the app context
    if not os.getenv('COINGLASS_API_KEY'):
        print("COINGLASS_API_KEY environment variable not set. Exiting.")
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(APITestCase)
        unittest.TextTestRunner(verbosity=2).run(suite)
