import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    data.dropna(inplace=True)
    # Convert date column to datetime object
    data['date'] = pd.to_datetime(data['date'])
    # Extract year, month, day as separate features
    data['Year'] = data['date'].dt.year
    data['Month'] = data['date'].dt.month
    data['Day'] = data['date'].dt.day
    # Optionally, drop the original date column if no longer needed
    data.drop('date', axis=1, inplace=True)
    return data

def feature_engineering(data):
    # Example: Creating a new feature that combines bathrooms and bedrooms
    data['BathBedRatio'] = data['bathrooms'] / data['bedrooms']
    return data

def split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def predict_new_data(model, new_data):
    new_price = model.predict(new_data)
    return new_price

# Example usage within this script (if needed)
if __name__ == "__main__":
    data = load_and_clean_data("data.csv")
    data = feature_engineering(data)
    X_train, X_test, y_train, y_test = split_data(data, 'price')
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f'MSE: {mse}, R^2: {r2}')

