import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import yfinance as yf

# Function to fetch historical stock data using yfinance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

# Example: Get historical stock data for Apple Inc. from Yahoo Finance
ticker = 'AAPL'
start_date = '2023-01-01'
end_date = '2024-01-01'

# Fetch historical stock data
stock_data = get_stock_data(ticker, start_date, end_date)

# Plot the historical stock data
plt.plot(stock_data)
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# Split the data into training and testing sets
train_size = int(len(stock_data) * 0.8)
train, test = stock_data[:train_size], stock_data[train_size:]

# Fit ARIMA model
order = (1, 1, 1)  # Example order, you may need to adjust this
model = ARIMA(train, order=order)
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

# Evaluate the model
rmse = sqrt(mean_squared_error(test, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the predictions against the actual values
plt.plot(train, label='Training Data')
plt.plot(test, label='Actual Data')
plt.plot(predictions, label='Predictions')
plt.title(f'ARIMA Stock Price Forecasting for {ticker}')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
