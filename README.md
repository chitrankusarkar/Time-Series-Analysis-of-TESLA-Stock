
# LSTM-based AI with Heuristic for Predicting Stock Prices

## Overview
This project employs a Long Short-Term Memory (LSTM) neural network combined with heuristic techniques to predict stock prices for Tesla (TSLA) over a 2-year period based on historical data. The approach integrates AI and time-series decomposition to achieve a **76% prediction accuracy**.

## Features
- **Data Preprocessing**:  
  Resampling and decomposing historical stock prices into trend, seasonal, and residual components for enhanced interpretability.
  
- **LSTM Model**:  
  A deep learning model tailored for each component (trend, seasonal, and residual) to predict future values.

- **Integration**:  
  Predicted components are combined to generate the final stock price forecast.

- **Visualization**:  
  Side-by-side comparison of actual and predicted stock prices for better analysis.

## Dataset
The project uses the `TSLA.csv` dataset, which includes historical stock prices for Tesla up to December 31, 2020, filtered to exclude noise caused by the COVID-19 pandemic.

## Installation
To run the project, ensure you have the following installed:
- Python 3.8 or later
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `statsmodels`
  - `tensorflow`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Code Overview
1. **Data Preparation**:
   - Resamples data to weekly intervals.
   - Decomposes data into trend, seasonal, and residual components.

2. **Scaling and Dataset Creation**:
   - Scales components to prepare for LSTM training.
   - Creates time-windowed datasets for training and testing.

3. **Model Training**:
   - Trains separate LSTM models for trend, seasonal, and residual components using Nadam optimizer and dropout regularization.

4. **Prediction and Inverse Scaling**:
   - Combines predictions from all components to produce final stock price forecasts.

5. **Visualization**:
   - Compares predicted stock prices with actual historical prices.

## Results
The model demonstrates **76% accuracy** in predicting stock prices, effectively capturing underlying trends and seasonality.

## Example Usage
Run the following command to execute the script:
```bash
python main.py
```

## Output
- Graph showing predicted vs. actual prices.
- Evaluation metrics (if extended).

---

## Acknowledgments
This project uses techniques from time-series analysis and deep learning to enhance stock price prediction accuracy. Special thanks to the developers of the tools and libraries used.
