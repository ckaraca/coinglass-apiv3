import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

from coinglass_api.api import CoinglassAPI


def load_config():
    with open('config.json', 'r') as config_file:
        return json.load(config_file)
    

"""
def get_coinglass_data(url, api_key, symbol="BTC"):
    headers = {
        "accept": "application/json",
        "coinglassSecret": api_key
    }
    params = {"symbol": symbol}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()['data']

def create_plots(cem):
    plt.figure(figsize=(16, 14))
    sns.set_style("whitegrid")

    # 1. Bar plot of Open Interest by Exchange
    plt.subplot(2, 2, 1)
    cem_sorted = cem.sort_values('openInterest', ascending=False)
    sns.barplot(x='openInterest', y='exchangeName', data=cem_sorted)
    plt.title('Open Interest by Exchange')
    plt.xlabel('Open Interest')
    plt.ylabel('Exchange')

    # 2. Scatter plot of Open Interest vs Volume
    plt.subplot(2, 2, 2)
    plt.scatter(cem['openInterest'], cem['volUsd'])
    plt.title('Open Interest vs Volume')
    plt.xlabel('Open Interest')
    plt.ylabel('Volume (USD)')
    for i, txt in enumerate(cem['exchangeName']):
        plt.annotate(txt, (cem['openInterest'].iloc[i], cem['volUsd'].iloc[i]))

    # 3. Heatmap of Correlation between numerical columns
    plt.subplot(2, 2, 3)
    numeric_cols = cem.select_dtypes(include=[np.number]).columns
    correlation = cem[numeric_cols].corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numerical Columns')

    # 4. Bar plot of 24h Change by Exchange
    plt.subplot(2, 2, 4)
    cem_sorted_change = cem.sort_values('h24Change', ascending=True)
    sns.barplot(x='h24Change', y='exchangeName', data=cem_sorted_change)
    plt.title('24h Change by Exchange')
    plt.xlabel('24h Change (%)')
    plt.ylabel('Exchange')

    plt.tight_layout()
    plt.savefig('coinglass_data_plot.png', dpi=300)
    print("Plot has been saved as 'coinglass_data_plot.png'")

def main():
    try:
        config = load_config()
        data = get_coinglass_data(config['coinglass_url'], config['coinglass_apikey'])
        
        cem = pd.DataFrame(data)
        print(cem.head())
        print(cem.shape)

        create_plots(cem)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Exception details:")
        print(sys.exc_info())

    finally:
        plt.close()  # Close the plot to free up memory

"""
def main():
    try:
        config = load_config()
        cg = CoinglassAPI(config['coinglass_apikey'])
        cem = cg.open_interest_exchange_list(symbol="BTC")
        print(cem.head())
        print(cem.info())
        print(cem.shape)
        
        # cem1 = cg.open_interest_history("h1", symbol="BTC")
        # print(cem1.head())
        # print(cem1.info())
        # print(cem1.shape)
        
        # get open interest history for the last 24 hours
        cem2 = cg.open_interest_history("1d", symbol="BTC")
        print(cem2.head())
        print(cem2.info())
        

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Exception details:")
        print(sys.exc_info())

    finally:
        plt.close()  # Close the plot to free up memory

if __name__ == "__main__":
    main()
    
