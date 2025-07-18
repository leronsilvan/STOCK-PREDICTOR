import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def predict_stock_with_features(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        st.error("No data found. Please check the symbol or date range.")
        return None, None

    # Feature Engineering
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Target'] = df['Close'].shift(-5)

    df.dropna(inplace=True)

    X = df[['Close', 'MA5', 'MA10', 'Daily_Return']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)

    return model, results, mse, mae, r2


# 🔵 Streamlit UI
st.title("📈 Stock Price Prediction App")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

if st.button("Predict"):
    model, results, mse, mae, r2 = predict_stock_with_features(symbol, start_date, end_date)

    if results is not None:
        st.subheader("📊 Model Evaluation")
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**Mean Absolute Error:** {mae:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        st.subheader("📈 Actual vs Predicted Closing Prices")
        st.line_chart(results)

        st.subheader("📋 Last 5 Predictions")
        st.dataframe(results.tail())