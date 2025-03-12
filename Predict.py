import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Fetch historical stock data from Yahoo Finance
stock_data = yf.download('AAPL', start='2010-01-01', end='2023-12-31')
stock_data.to_csv('stock_prices.csv')
print("Data saved to stock_prices.csv")

# Load the data from the saved CSV file
data = pd.read_csv('stock_prices.csv')

# Ensure the 'Date' column is parsed as a datetime object and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Round stock prices to 1 decimal place
price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data[price_columns] = data[price_columns].round(1)

# Feature Engineering: Create a lagged feature for prediction
data['Previous_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

# Define features and target variable
X = data[['Previous_Close']]
y = data['Close']

# Split the data into training and testing sets (no shuffling to maintain time series order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the models
linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()

linear_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Make predictions using the trained models
linear_predictions = linear_model.predict(X_test)
decision_tree_predictions = decision_tree_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate the models using R-squared scores
linear_r2 = r2_score(y_test, linear_predictions)
decision_tree_r2 = r2_score(y_test, decision_tree_predictions)
random_forest_r2 = r2_score(y_test, random_forest_predictions)

print("\nModel Performance:")
print(f"Linear Regression R-squared: {linear_r2:.4f}")
print(f"Decision Tree R-squared: {decision_tree_r2:.4f}")
print(f"Random Forest R-squared: {random_forest_r2:.4f}")

# Visualization of actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', color='black')
plt.plot(y_test.index, linear_predictions, label='Linear Regression', alpha=0.7)
plt.plot(y_test.index, decision_tree_predictions, label='Decision Tree', alpha=0.7)
plt.plot(y_test.index, random_forest_predictions, label='Random Forest', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction Using Different Models')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Predict future price using the latest available data
last_price = np.array([data['Previous_Close'].iloc[-1]]).reshape(-1, 1)

predicted_price_linear = linear_model.predict(last_price)
predicted_price_decision_tree = decision_tree_model.predict(last_price)
predicted_price_random_forest = random_forest_model.predict(last_price)

print(f"\nPredicted Next Day Price (Linear Regression): {predicted_price_linear[0]:.2f}")
print(f"Predicted Next Day Price (Decision Tree): {predicted_price_decision_tree[0]:.2f}")
print(f"Predicted Next Day Price (Random Forest): {predicted_price_random_forest[0]:.2f}")
