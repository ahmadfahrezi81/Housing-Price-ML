import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import joblib

# Obtaining the dataset
dataset = pd.read_csv("House Pricing.csv")

# Drop records that have null values
new_dataset = dataset.dropna()

# Categorizing dataset
obj = dataset.dtypes == "object"
object_cols = list(obj[obj].index)

OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = [
    f"{col}_{val}"
    for col, cats in zip(object_cols, OH_encoder.categories_)
    for val in cats
]
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Split data to training & testing sets
X = df_final.drop(["price"], axis=1)
Y = df_final["price"]
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)

# Linear Regression model
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
r2_LR = r2_score(Y_valid, Y_pred)

# Support Vector Machine model
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_SVR = model_SVR.predict(X_valid)
r2_SVR = r2_score(Y_valid, Y_pred_SVR)

# Polynomial Regression model
degree = 3
poly = PolynomialFeatures(degree=degree)
X_poly_train = poly.fit_transform(X_train)
X_poly_valid = poly.transform(X_valid)
model_PR = LinearRegression()
model_PR.fit(X_poly_train, Y_train)
Y_pred_PR = model_PR.predict(X_poly_valid)
r2_PR = r2_score(Y_valid, Y_pred_PR)

# SAVE THE MODEL
joblib.dump(model_SVR, "svr_model.sav")
