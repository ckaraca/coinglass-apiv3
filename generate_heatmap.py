import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from coinglass_api.api import CoinglassAPIv3

# Obtener API key de la variable de entorno
api_key = os.environ.get("COINGLASS_API_KEY")

# Verificar si la clave API está configurada
if not api_key:
    print("Error: la variable de entorno COINGLASS_API_KEY no está configurada.")
else:
    try:
        # Instanciar el cliente de la API de Coinglass
        cg = CoinglassAPIv3(coinglass_secret=api_key)

        # Definir parámetros por defecto
        exchange = "Binance"
        symbol = "BTCUSDT"
        range = "3d"  # Puedes cambiar el rango según sea necesario

        # Obtener datos del mapa de calor de liquidación
        liquidation_data = cg.liquidation_heatmap_model2(exchange=exchange, symbol=symbol, range=range)

        # Extraer datos para el mapa de calor
        price_levels = liquidation_data['price_levels']
        liquidations = liquidation_data['liquidations']
        ohlc = liquidation_data['ohlc']

        # Crear una tabla dinámica para el mapa de calor
        heatmap_data = liquidations.pivot_table(index='price', columns='time', values='liquidation_value', fill_value=0)

        # Crear una figura y un mapa de calor
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap="YlOrRd")

        # Establecer título y etiquetas de los ejes
        plt.title(f"Mapa de Calor de Liquidación para {symbol} en {exchange} ({range})")
        plt.xlabel("Tiempo")
        plt.ylabel("Nivel de Precio")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Guardar la gráfica
        output_path = f"images/{exchange}_{symbol}_liquidation_heatmap_{range}.png"
        plt.savefig(output_path)
        print(f"Mapa de calor de liquidación guardado en {output_path}")

    except Exception as e:
        print(f"Ocurrió un error: {e}")