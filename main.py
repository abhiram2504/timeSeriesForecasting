import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

# Fetch historical stock data
ticker_symbol = "AAPL"
data = yf.download(ticker_symbol, start="2023-01-01", end="2024-01-01")

# Extract the 'Close' prices as the variable to forecast
df = data[['Close']].reset_index()

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# Plot the historical stock prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Historical Prices')
plt.title(f"{ticker_symbol} Stock Prices")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Create an ARIMA model
order = (5, 1, 0)  # Example order, adjust as needed
model = ARIMA(train['Close'], order=order)

# Fit the model
fitted_model = model.fit()

# Forecast future values
forecast, _, _ = fitted_model.forecast(steps=len(test))

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(test['Date'], test['Close'], label='Actual')
plt.plot(test['Date'], forecast, label='Forecast')
plt.title(f"{ticker_symbol} Stock Prices - Actual vs. Forecast")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
