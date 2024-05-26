import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Obtaining the dataset
dataset = pd.read_csv("House Pricing.csv")
print(dataset.head(5))

# Learn the dataset
sns.pairplot(dataset)
# plt.show()

# check the correlation
dataset.corr(numeric_only=True)

# visualise the correlation
# plt.figure(figsize=(10, 10))
sns.heatmap(dataset.corr(numeric_only=True), annot=True)

# Drop records that have null values (Empty records are very less).
new_dataset = dataset.dropna()

# Checking features which have null values in the new dataframe (if there are still any).

new_dataset.isnull().sum()

# Use sklearn.preprocessing to import OneHotEncoder


s = new_dataset.dtypes == "object"
object_cols = list(s[s].index)
# print("Categorical variables:")
# print(object_cols)
# print("No. of. categorical features: ", len(object_cols))

# Now we have list of all the features, can apply OneHotEncoding to the list

OH_encoder = OneHotEncoder(
    sparse_output=False
)  # Note that: `sparse_output` has been renamed to `sparse`
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
# Manually create column names according to encoder's `categories_` attribute
OH_cols.columns = [
    f"{col}_{val}"
    for col, cats in zip(object_cols, OH_encoder.categories_)
    for val in cats
]
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Categorizing dataset
# Object type
obj = dataset.dtypes == "object"
object_cols = list(obj[obj].index)
# print("Categorical variables:", len(object_cols))
# Integer type
int_ = dataset.dtypes == "int"
num_cols = list(int_[int_].index)
# print("Integer variables:", len(num_cols))
# Float type
fl = dataset.dtypes == "float"
fl_cols = list(fl[fl].index)
# print("Float variables:", len(fl_cols))
# Long type
long1 = dataset.dtypes == "int64"
long_cols = list(long1[long1].index)
# print("Long variables:", len(long_cols))

# plotting the Bar plot
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].nunique())

# plt.figure(figsize=(10, 6))
# plt.title("Unique values in categorical features")
# plt.xticks(rotation=90)
# sns.barplot(x=object_cols, y=unique_values)

# plt.figure(figsize=(18, 36))
# plt.suptitle("Categorical values", fontsize=20)
# plt.subplots_adjust(hspace=0.5)

index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    ax = plt.subplot(11, 4, index)
    ax.set_title(col)
    plt.xticks(rotation=90)
    sns.barplot(x=y.index, y=y)
    index += 1

# Replace SalePrice missing values with mean values
dataset["price"] = dataset["price"].fillna(dataset["price"].mean())

# Drop the records with null values (empty records are very less).
new_dataset = dataset.dropna()

# Check features with null values in  new dataframe.
new_dataset.isnull().sum()

# Split data to training & testing sets


X = df_final.drop(["price"], axis=1)
Y = df_final["price"]
print(df_final.corr()["price"].sort_values())
X = X.drop(
    [
        "airconditioning_no",
        "prefarea_no",
        "mainroad_no",
        "furnishingstatus_unfurnished",
        "guestroom_no",
        "basement_no",
        "hotwaterheating_no",
        "furnishingstatus_semi-furnished",
        "hotwaterheating_yes",
        "basement_yes",
        "furnishingstatus_furnished",
    ],
    axis=1,
)
# Split training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)


LR = LinearRegression()
LR.fit(X_train, Y_train)

# new cell
Y_pred = LR.predict(X_valid)  # Changed Y_valid to X_valid

# new cell
MAE_LM = mean_absolute_error(Y_valid, Y_pred)
MSE_LM = mean_squared_error(Y_valid, Y_pred)
RMSE_LM = np.sqrt(MSE_LM)  # Corrected typo here

# print("MAE:", MAE_LM, "\nMSE:", MSE_LM, "\nRMSE:", RMSE_LM)


# To calculate loss we will use module mean_absolute_percentage_error


# Model 1: Linear Regression


model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
# The Intercept
# print("\nIntercept: \n", model_LR.intercept_)
# print("\nIntercept: \n", model_LR.coef_)
# Evaluation of: Linear Regression
Y_pred = model_LR.predict(X_valid)
mse = mean_squared_error(Y_valid, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_valid, Y_pred)
# print(
#     "Linear Regression Evaluation:\nMSE: {}\nRMSE: {}\nR-squared: {}".format(
#         mse, rmse, r2
#     )
# )
# Calculate MSE with sklearn
mse = mean_squared_error(Y_valid, Y_pred)
# print("MSE using sklearn: ", mse)
plt.scatter(Y_valid, Y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual Prices vs Predicted prices")
plt.show()

# Model 2: Support Vector machine


model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_predSV = model_SVR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_predSV))

# Evaluation of Support Vector Machine
Y_pred = model_SVR.predict(X_valid)
mse_svr = mean_squared_error(Y_valid, Y_pred)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(Y_valid, Y_pred)
print(
    "Support Vector Machine Evaluation:\nMSE: {}\nRMSE: {}\nR-squared: {}".format(
        mse_svr, rmse_svr, r2_svr
    )
)
# The Intercept
print("\nIntercept: \n", model_SVR.intercept_)
# Add mean squared error
mse = mean_squared_error(Y_valid, Y_pred)
print("MSE using sklearn: ", mse)
plt.scatter(Y_valid, Y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual Prices vs Predicted prices")
plt.show()

# Model: Polynomial Regression
degrees = [2, 3, 4, 5, 6]
MAPE_PR = []
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_valid = poly.transform(X_valid)
    model_PR = LinearRegression()
    model_PR.fit(X_poly_train, Y_train)
    Y_pred_PR = model_PR.predict(X_poly_valid)
    MAPE_PR.append(mean_absolute_percentage_error(Y_valid, Y_pred_PR))

print(MAPE_PR)

# Evaluation of: Polynomial Regression
Y_poly_pred = model_PR.predict(X_poly_valid)
mse_poly = mean_squared_error(Y_valid, Y_poly_pred)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(Y_valid, Y_poly_pred)
print(
    "Polynomial Regression Evaluation:\nMSE: {}\nRMSE: {}\nR-squared: {}".format(
        mse_poly, rmse_poly, r2_poly
    )
)

# Check Intercept
print("\nIntercept: \n", model_PR.intercept_)
# Add mean squared error
mse = mean_squared_error(Y_valid, Y_poly_pred)
print("MSE using sklearn: ", mse)
plt.scatter(Y_valid, Y_poly_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual Prices vs Predicted prices")
plt.show()


# SAVE THE MODEL HERE
joblib.dump(model_SVR, "svr_model.sav")
