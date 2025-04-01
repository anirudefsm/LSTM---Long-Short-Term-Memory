# LSTM-Based Stock Price Prediction for HUL

## Project Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices for Hindustan Unilever Limited (HUL). The model leverages historical stock price data to capture sequential dependencies and long-term trends, providing insights into the potential of deep learning for financial forecasting.

## Contributors
- Anirudh Gupta
  
## Problem Statement
Stock market prediction is challenging due to the highly volatile and non-linear nature of financial data. Traditional statistical models often struggle to capture these complexities. This project addresses this challenge by using LSTM networks, which excel at modeling sequential data, to enhance forecasting accuracy.

## Features
- **Data Collection**: Utilizes Yahoo Finance (`yfinance`) to fetch historical stock price data.
- **Data Preprocessing**: Includes normalization, sequence transformation, and train-test splitting.
- **Model Architecture**: 
  - 3 LSTM layers (50 units each) with Dropout (0.2) to prevent overfitting.
  - 25-unit ReLU layer + 1-unit linear output.
  - Optimizer: Adam (`learning_rate = 0.001`).
  - Loss Function: Mean Squared Error (MSE), with Mean Absolute Error (MAE) as an additional metric.
- **Training**: 
  - Epochs: 10
  - Batch Size: 32
  - Validation: Monitored using test data.
- **Performance Metrics**:
  - Final MSE: 0.0002 (training), 0.0020 (validation).
  - Final MAE: 0.0107 (training), 0.0418 (validation).
  - MAPE: 4.96% (test set), implying **95.04% accuracy**.

## Key Observations
1. **High Accuracy**: The model achieves **95.04% accuracy** on test data.
2. **Limitations**:
   - Struggles with sudden market shocks (e.g., geopolitical events).
   - Excludes external factors (e.g., news sentiment, macroeconomic indicators).

## Managerial Insights
- **Portfolio Optimization**: Use LSTM predictions for dynamic rebalancing of HUL-linked portfolios.
- **Risk Management**: Integrate LSTM outputs into stress-testing models for liquidity risk assessment.
- **Algorithmic Trading**: Develop LSTM-based trading bots to exploit predicted short-term trends.
- **Regulatory Compliance**: Use predictions to simulate Basel III scenarios for capital adequacy.
- **Retail Investment Tools**: Offer AI-powered advisory platforms with real-time HUL stock forecasts.

## Project Structure
1. **Importing Libraries**
2. **Data Collection & Preprocessing**
3. **Model Development**
4. **Training & Evaluation**
5. **Prediction & Visualization**

## Libraries Used
- **Data Handling & Processing**: `numpy`, `pandas`
- **Data Visualization**: `matplotlib.pyplot`, `seaborn`
- **Stock Data Retrieval**: `yfinance`
- **Machine Learning & Deep Learning**: `tensorflow.keras` (`Sequential`, `LSTM`, `Dense`, `Dropout`)
- **Performance Metrics**: `mean_squared_error`, `mean_absolute_error`
- **Data Scaling**: `MinMaxScaler` (from `sklearn.preprocessing`)

## Conclusion
The LSTM model achieves **95.04% accuracy** in predicting HUL stock trends, demonstrating the viability of deep learning for financial forecasting. This study provides insights into the practical applications of LSTM-based models in financial markets, showcasing their potential to improve decision-making for traders, investors, and financial analysts.

## How to Use
1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `055003_LSTM_Stock_Price_Prediction_HUL.ipynb` to execute the project.

