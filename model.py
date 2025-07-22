import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])  # predict next day's closing price
    return np.array(X), np.array(y)


def predict_stock_with_time_series(symbol, start_date, end_date, window_size=60):
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        st.error("No data found. Please check the symbol or date range.")
        return None, None, None, None, None

    df = df[['Close']].dropna()

    # Scale the closing price
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create time series sequences
    X, y = create_sequences(scaled_data, window_size)

    # Train-test split (no shuffle for time series)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for Linear Regression (2D)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Inverse transform to original scale
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Evaluation
    mse = mean_squared_error(y_test_inv, predictions_inv)
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    r2 = r2_score(y_test_inv, predictions_inv)

    # Create results DataFrame
    results = pd.DataFrame({
        'Actual': y_test_inv.flatten(),
        'Predicted': predictions_inv.flatten()
    }, index=df.index[-len(y_test):])

    return model, results, mse, mae, r2


st.title("📈 Stock Price Time Series Predictor")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

if st.button("Predict"):
    model, results, mse, mae, r2 = predict_stock_with_time_series(symbol, start_date, end_date)

    if results is not None:
        st.subheader("📊 Model Evaluation")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        st.subheader("📈 Actual vs Predicted Closing Prices")
        st.line_chart(results)

        st.subheader("📋 Last 5 Predictions")
        st.dataframe(results.tail())
