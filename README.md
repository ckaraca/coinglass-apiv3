# Coinglass API Wrapper

This project is a Flask-based API wrapper for the Coinglass API, providing a structured and easy-to-use interface to access cryptocurrency derivatives data.

**&copy; 2024 Universal Business Technology. All Rights Reserved.**

---

## Project Overview

This API acts as an intermediary between a web application and the official Coinglass API. It is designed to be deployed on a serverless platform like Vercel. The main purposes of this wrapper are:

*   **Secure API Key Management:** The Coinglass API key is stored securely as an environment variable on the server, never exposing it to the client-side application.
*   **Data Transformation:** The API transforms data from the Coinglass Python library (which uses pandas DataFrames) into a standard JSON format that is easily consumable by web applications.
*   **Simplified Endpoints:** It provides a clean and well-documented set of endpoints, abstracting the complexities of the underlying `coinglass_api` library.
*   **Error Handling:** Implements robust error handling to provide clear feedback when an API call fails.

## Available Endpoints

The following endpoints are available. For endpoints with parameters, you can pass them as query strings in the URL (e.g., `/liquidation/history?symbol=ETHUSDT&interval=h4`).

### General

*   **`/`**
    *   **Description:** Provides a welcome message and a list of all available endpoints.
    *   **Method:** `GET`
    *   **Parameters:** None

### Liquidation Data

*   **`/liquidation/history`**
    *   **Description:** Retrieves historical liquidation data for a specific trading pair.
    *   **Method:** `GET`
    *   **Parameters:**
        *   `exchange` (string, optional, default: `Binance`)
        *   `symbol` (string, optional, default: `BTCUSDT`)
        *   `interval` (string, optional, default: `h1`)
        *   `limit` (integer, optional, default: `100`)

*   **`/liquidation/aggregated-history`**
    *   **Description:** Retrieves aggregated historical liquidation data for a specific coin.
    *   **Method:** `GET`
    *   **Parameters:**
        *   `symbol` (string, optional, default: `BTC`)
        *   `interval` (string, optional, default: `h1`)
        *   `limit` (integer, optional, default: `100`)

*   **`/liquidation/coin-list`**
    *   **Description:** Fetches liquidation data for all coins on a specified exchange.
    *   **Method:** `GET`
    *   **Parameters:**
        *   `ex` (string, optional, default: `Binance`)

*   **`/liquidation/exchange-list`**
    *   **Description:** Fetches liquidation data for a coin across all exchanges.
    *   **Method:** `GET`
    *   **Parameters:**
        *   `symbol` (string, optional, default: `BTC`)
        *   `range` (string, optional, default: `1h`)

### Long/Short Ratio

*   **`/long-short-ratio/global-history`**
    *   **Description:** Retrieves the global long/short account ratio for a trading pair.
    *   **Method:** `GET`
    *   **Parameters:** (Same as `/liquidation/history`)

*   **`/long-short-ratio/top-account-history`**
    *   **Description:** Retrieves the long/short ratio history for top accounts.
    *   **Method:** `GET`
    *   **Parameters:** (Same as `/liquidation/history`)

*   **`/long-short-ratio/top-position-history`**
    *   **Description:** Retrieves the long/short ratio history for top positions.
    *   **Method:** `GET`
    *   **Parameters:** (Same as `/liquidation/history`)

### Funding Rate

*   **`/funding-rate/ohlc-history`**
    *   **Description:** Retrieves OHLC history for funding rates.
    *   **Method:** `GET`
    *   **Parameters:** (Same as `/liquidation/history`)

### Open Interest

*   **`/open-interest/ohlc-history`**
    *   **Description:** Retrieves OHLC history for open interest.
    *   **Method:** `GET`
    *   **Parameters:** (Same as `/liquidation/history`)

### Global and Spot Markets

*   **`/global/coins-markets`**
    *   **Description:** Retrieves performance information for all available coins.
    *   **Method:** `GET`
    *   **Parameters:**
        *   `exchanges` (string, optional, default: `Binance,OKX`)
        *   `page_num` (integer, optional, default: 1)
        *   `page_size` (integer, optional, default: 100)

*   **`/global/pairs-markets`**
    *   **Description:** Retrieves performance information for all pairs of a specific coin.
    *   **Method:** `GET`
    *   **Parameters:**
        *   `symbol` (string, optional, default: `BTC`)

*   **`/spot/pairs-markets`**
    *   **Description:** Retrieves performance information for all pairs of a specific coin in the Spot market.
    *   **Method:** `GET`
    *   **Parameters:**
        *   `symbol` (string, optional, default: `BTC`)

### Indicators

*   **`/indicator/bitcoin-bubble-index`**
    *   **Description:** Fetches the Bitcoin Bubble Index.
    *   **Method:** `GET`
    *   **Parameters:** None

*   **`/indicator/fear-greed-index`**
    *   **Description:** Fetches the Fear & Greed Index.
    *   **Method:** `GET`
    *   **Parameters:** None

*   **`/indicator/ahr999-index`**
    *   **Description:** Fetches the AHR999 Index.
    *   **Method:** `GET`
    *   **Parameters:** None

*   **`/indicator/two-year-ma-multiplier`**
    *   **Description:** Fetches the Two-Year MA Multiplier.
    *   **Method:** `GET`
    *   **Parameters:** None

*   **`/indicator/puell-multiple`**
    *   **Description:** Fetches the Puell Multiple.
    *   **Method:** `GET`
    *   **Parameters:** None

---

## Changelog

*   **Initial Commit:** Base Flask application setup.
*   **API Refactoring (1):**
    *   Corrected the initialization of the `CoinglassAPIv3` class.
    *   Added a helper function (`dataframe_to_json`) to correctly serialize pandas DataFrame objects to JSON.
    *   Implemented a basic set of endpoints for core functionalities.
    *   Added robust error handling for all endpoints.
*   **API Refactoring (2):**
    *   Expanded the API with additional endpoints for liquidation and long/short ratio data.
    *   Refined the code structure for better readability and maintenance.
    *   Added a comprehensive test suite (`api_test.py`) to simulate API calls and verify endpoint logic using mocks.
*   **Documentation:**
    *   Created a new, detailed `README.md` file.
    *   Added a universal copyright notice.
    *   Documented all available endpoints, including their parameters and default values.
    *   Included a changelog to track project development.
*   **Feature Expansion (1): Global & Spot Markets**
    *   Added endpoints to retrieve market-wide data for both futures (`/global/coins-markets`, `/global/pairs-markets`) and spot (`/spot/pairs-markets`).
    *   Updated the test suite to cover the new market endpoints.
*   **Feature Expansion (2): Advanced Bitcoin Indicators**
    *   Added endpoints for several key on-chain Bitcoin indicators:
        *   `/indicator/ahr999-index`
        *   `/indicator/two-year-ma-multiplier`
        *   `/indicator/puell-multiple`
    *   Updated the test suite to include these new indicator endpoints.

---

## Deployment

This application is intended for deployment on a platform like Vercel. To deploy:

1.  Push the code to a GitHub repository.
2.  Create a new project on Vercel and link it to the repository.
3.  Set the `COINGLASS_API_KEY` as an environment variable in the Vercel project settings.
4.  Vercel will automatically build and deploy the application.
