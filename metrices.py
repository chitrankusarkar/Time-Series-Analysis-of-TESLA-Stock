from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_trend_inv, test_predictions)
print(f"Mean Absolute Error (MAE): {mae}")
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_trend_inv, test_predictions)
print(f"Mean Squared Error (MSE): {mse}")
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")
# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test_trend_inv, test_predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
# Calculate simple accuracy percentage 
accuracy = np.mean(np.abs((y_test_trend_inv - test_predictions) / y_test_trend_inv) < 0.1)
print(f"Accuracy: {accuracy * 100:.2f}%")
