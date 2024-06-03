import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time


class HousePricePredictor:
    def __init__(self, csv_path):
        print("Loading data...")
        self.data = pd.read_csv(csv_path)
        print("Data loaded. Size: ", self.data.shape)

        self.clean_data()
        self.eda()
        self.hypothesis_testing()
        self.prepare_data()
        self.train_and_evaluate_model()
        self.save_model()

    def clean_data(self):
        print("Cleaning data...")
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data = self.data.drop(
            [
                "date",
                "street",
                "city",
                "statezip",
                "country",
                "yr_renovated",
                "sqft_basement",
            ],
            axis=1,
        )
        self.data = self.data.fillna(0)
        self.data = self.data.replace(
            [np.inf, -np.inf], np.nan
        )  # Replace infinite values with NaN
        self.data = self.data.dropna()  # Drop rows with NaN values if any remain

        # Adjusting year built to house age
        self.data["house_age"] = 2014 - self.data["yr_built"]
        self.data = self.data.drop("yr_built", axis=1)

        print("Data Cleaning Complete")

    def eda(self):
        print("Starting EDA...")
        print(self.data.describe())

        # Visualization: Distribution of prices
        sns.histplot(self.data["price"], kde=True)
        plt.title("Distribution of House Prices")
        plt.show(block=False)
        plt.pause(1)  # Keep it open for 1 second

        # Visualization: Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show(block=False)
        plt.pause(1)  # Keep it open for 1 second

        # Scatter plot for bedrooms vs. price
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="bedrooms", y="price", data=self.data)
        plt.title("Scatter plot of Bedrooms vs. Price")
        plt.show(block=False)
        plt.pause(1)  # Keep it open for 1 second

        print("EDA Complete")

    def hypothesis_testing(self):
        print("Starting Hypothesis Testing...")
        waterfront = self.data[self.data["waterfront"] == 1]["price"]
        non_waterfront = self.data[self.data["waterfront"] == 0]["price"]

        t_stat, p_value = stats.ttest_ind(waterfront, non_waterfront)
        print(
            f"T-test for price difference between waterfront and non-waterfront properties: p-value = {p_value}"
        )

        if p_value < 0.05:
            print(
                "Reject the null hypothesis: There is a significant difference in prices."
            )
        else:
            print(
                "Fail to reject the null hypothesis: No significant difference found in prices."
            )

        print("Hypothesis Testing Complete")

    def prepare_data(self):
        print("Preparing data for modeling...")

        # Preparing your features and target
        self.X = self.data.drop("price", axis=1)
        self.y = self.data["price"]

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print(
            "Data preparation complete. Sizes: ", self.X_train.shape, self.X_test.shape
        )

    def train_and_evaluate_model(self):
        print("Training the model...")
        self.model = Ridge(alpha=1.0)
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        # Additional check to see the coefficients
        coefficients = pd.DataFrame(
            self.model.coef_,
            self.data.drop("price", axis=1).columns,
            columns=["Coefficient"],
        )
        print(coefficients)

    def save_model(self):
        print("Saving the model...")
        joblib.dump(
            {"model": self.model, "scaler": self.scaler}, "house_price_model.sav"
        )
        print("Model saved to house_price_model.sav")

    def predict_price(self, features):
        feature_names = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "sqft_above",
            "house_age",  # Change from yr_built to house_age
        ]
        features_df = pd.DataFrame([features], columns=feature_names)

        # Use existing scaler to transform features
        features_scaled = self.scaler.transform(features_df)

        return self.model.predict(features_scaled)[0]


# Example usage
start_time = time.time()

print("Starting prediction process...")
predictor = HousePricePredictor("data.csv")

# Example prediction for a house with proper adjustment to match 10 columns
features = [
    3.0,
    1.5,
    1340,
    7912,
    1.5,
    0,
    0,
    3,
    1340,
    59,
]  # Last value is house_age (2014 - 1955)
predicted_price = predictor.predict_price(features)
print(f"Predicted Price: ${predicted_price:.2f}")

end_time = time.time()
print(f"Total Runtime: {end_time - start_time} seconds")
