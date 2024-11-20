import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Nadam

# Load data
data = pd.read_csv('TSLA.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filter data to include only up to 2020 to avoid covid effect(noise)
data = data[data.index <= '2020-12-31']

# Resample the data to weekly frequency, using the mean of the daily adjusted close prices
data = data.resample('W').mean()

# Perform seasonal decomposition
result = seasonal_decompose(data['Adj Close'], model='multiplicative', period=52)
data['trend'] = result.trend
data['seasonal'] = result.seasonal
data['residual'] = result.resid

# Drop NaN values created by the decomposition
data.dropna(inplace=True)

# Scale data
scaler = MinMaxScaler()
scaled_trend = scaler.fit_transform(data['trend'].values.reshape(-1, 1))
scaled_seasonal = scaler.fit_transform(data['seasonal'].values.reshape(-1, 1))
scaled_residual = scaler.fit_transform(data['residual'].values.reshape(-1, 1))

# Split the data
train_size = len(scaled_trend) - 104  # Last 24 months for testing
train_trend, test_trend = scaled_trend[:train_size], scaled_trend[train_size:]
train_seasonal, test_seasonal = scaled_seasonal[:train_size], scaled_seasonal[train_size:]
train_residual, test_residual = scaled_residual[:train_size], scaled_residual[train_size:]

# Prepare dataset
n_input = 3  # number of previous weeks to use for prediction
n_output = 1  # predicting the next week

def create_dataset(data, n_input, n_output):
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i+n_input])
        y.append(data[i+n_input:i+n_input+n_output])
    return np.array(X), np.array(y)

X_train_trend, y_train_trend = create_dataset(train_trend, n_input, n_output)
X_test_trend, y_test_trend = create_dataset(test_trend, n_input, n_output)

X_train_seasonal, y_train_seasonal = create_dataset(train_seasonal, n_input, n_output)
X_test_seasonal, y_test_seasonal = create_dataset(test_seasonal, n_input, n_output)

X_train_residual, y_train_residual = create_dataset(train_residual, n_input, n_output)
X_test_residual, y_test_residual = create_dataset(test_residual, n_input, n_output)

#  y_train and y_test are reshaped correctly for inverse scaling
y_train_trend = y_train_trend.reshape(-1, 1)
y_test_trend = y_test_trend.reshape(-1, 1)

y_train_seasonal = y_train_seasonal.reshape(-1, 1)
y_test_seasonal = y_test_seasonal.reshape(-1, 1)

y_train_residual = y_train_residual.reshape(-1, 1)
y_test_residual = y_test_residual.reshape(-1, 1)

# Define LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(150, return_sequences=True, input_shape=(n_input, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(LSTM(150, return_sequences=True))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(LSTM(150))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(n_output))
    model.compile(optimizer=Nadam(), loss='mae')
    return model

# Train models for each component
model_trend = create_lstm_model()
history_trend = model_trend.fit(X_train_trend, y_train_trend, epochs=100, validation_data=(X_test_trend, y_test_trend), verbose=1)

model_seasonal = create_lstm_model()
history_seasonal = model_seasonal.fit(X_train_seasonal, y_train_seasonal, epochs=100, validation_data=(X_test_seasonal, y_test_seasonal), verbose=1)

model_residual = create_lstm_model()
history_residual = model_residual.fit(X_train_residual, y_train_residual, epochs=100, validation_data=(X_test_residual, y_test_residual), verbose=1)

# Make predictions
train_predictions_trend = model_trend.predict(X_train_trend)
test_predictions_trend = model_trend.predict(X_test_trend)

train_predictions_seasonal = model_seasonal.predict(X_train_seasonal)
test_predictions_seasonal = model_seasonal.predict(X_test_seasonal)

train_predictions_residual = model_residual.predict(X_train_residual)
test_predictions_residual = model_residual.predict(X_test_residual)

# Inverse transform the predictions and actual values to original scale
trend_scaler = MinMaxScaler()
seasonal_scaler = MinMaxScaler()
residual_scaler = MinMaxScaler()


scaled_trend = trend_scaler.fit_transform(data['trend'].values.reshape(-1, 1)) 
scaled_seasonal = seasonal_scaler.fit_transform(data['seasonal'].values.reshape(-1, 1))
scaled_residual = residual_scaler.fit_transform(data['residual'].values.reshape(-1, 1))

train_predictions_trend = trend_scaler.inverse_transform(train_predictions_trend)
y_train_trend_inv = trend_scaler.inverse_transform(y_train_trend)
test_predictions_trend = trend_scaler.inverse_transform(test_predictions_trend)
y_test_trend_inv = trend_scaler.inverse_transform(y_test_trend)

train_predictions_seasonal = seasonal_scaler.inverse_transform(train_predictions_seasonal)
y_train_seasonal_inv = seasonal_scaler.inverse_transform(y_train_seasonal)
test_predictions_seasonal = seasonal_scaler.inverse_transform(test_predictions_seasonal)
y_test_seasonal_inv = seasonal_scaler.inverse_transform(y_test_seasonal)

train_predictions_residual = residual_scaler.inverse_transform(train_predictions_residual)
y_train_residual_inv = residual_scaler.inverse_transform(y_train_residual)
test_predictions_residual = residual_scaler.inverse_transform(test_predictions_residual)
y_test_residual_inv = residual_scaler.inverse_transform(y_test_residual)

# Combine the components to get final predictions
train_predictions = train_predictions_trend + train_predictions_seasonal + train_predictions_residual
test_predictions = test_predictions_trend + test_predictions_seasonal + test_predictions_residual

# Plot predicted vs actual values for the test set
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test_trend_inv):], data['Adj Close'][-len(y_test_trend_inv):], label='Actual Prices')
plt.plot(data.index[-len(y_test_trend_inv):], test_predictions, label='Predicted Prices')
plt.title('Predicted vs Actual Prices (up to 2020)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
