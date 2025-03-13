from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.pipeline import make_pipeline

df_og = pd.read_csv("cleaned_properties_info_new.csv")

df = df_og[["Property ID", "Furnishing", "Bathroom", "Tenant Preferred", "Availability", "Floor", "Facing", "Overlooking", "Balcony", "Ownership", "Property Type", "Latitude", "Longitude", "Address Region", "Price", "Floor Number", "Total Floors", "Rental Start Year", "Rental Start Month", "Area", "Garden/Park", "Main Road", "Pool"]]

# Drop "Unknown_Availability" entries (only around 5)
df = df[df["Availability"] != "Unknown_Availability"]

# Define a mapping for month abbreviations to numbers
month_mapping = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

# Function to compute "Days Until Available" using Rental Start Year/Month as the reference point
def compute_months_until_available(row):
    availability = row["Availability"].strip('"')  # Remove quotation marks

    if availability == "Immediately":
        return 0  # Already available

    try:
        parts = availability.split()
        avail_month = month_mapping[parts[1][:3]]  # Convert "Apr" to 4, "May" to 5, etc.
        avail_year = int("20" + parts[2])  # Convert "25" to 2025, "26" to 2026, etc.

        # Compute months difference
        year_diff = avail_year - row["Rental Start Year"]
        month_diff = avail_month - row["Rental Start Month"]
        
        return (year_diff * 12) + month_diff
    except:
        return None  # Handle unexpected cases

df["Months_Until_Available"] = df.apply(compute_months_until_available, axis=1)

# Drop original Availability column
#df.drop(columns=["Availability"], inplace=True)

# Convert ">10" values in Bathroom and Balcony to 11
# Ensure Bathroom and Balcony values are properly cleaned and converted
df["Bathroom"] = df["Bathroom"].astype(str).str.strip().replace({">10": 11, "> 10": 11}).astype(int)
df["Balcony"] = df["Balcony"].astype(str).str.strip().replace({">10": 11, "> 10": 11}).astype(int)

# Define categorical columns
one_hot_columns = ["Furnishing", "Tenant Preferred", "Facing", "Ownership", "Property Type", "Address Region"]

# Create one-hot encoded variables separately
#df_dummies = pd.get_dummies(df[one_hot_columns], prefix=one_hot_columns)

# Concatenate one-hot encoded columns while keeping the originals
#df = pd.concat([df, df_dummies], axis=1)

df = pd.get_dummies(df, columns=one_hot_columns)


# Identify all "Unknown_" dummies
unknown_cols = [col for col in df.columns if "Unknown_" in col]

# Drop only "Unknown_" dummy variables to keep one reference category (avoiding multicollinearity)
df = df.drop(columns=unknown_cols, errors="ignore")

# Convert all boolean (True/False) columns to integer (0/1)
bool_cols = df.select_dtypes(include=["bool"]).columns  # Select boolean columns
df[bool_cols] = df[bool_cols].astype(int)

# Drop original Overlooking column
#df.drop(columns=["Overlooking"], inplace=True)
df.to_csv('knn_data.csv', index=False)


X = df.drop(columns=["Property ID", "Price", "Availability", "Floor", "Overlooking", "Rental Start Month", "Rental Start Year"])  # Drop 'Property ID', irrelevent, and target column 'Price'
y = df["Price"]  # Set the target variable
X = X.dropna()
y = y[X.index]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = GridSearchCV(estimator=make_pipeline(StandardScaler(), KNeighborsRegressor()), param_grid={
    "kneighborsregressor__n_neighbors": range(1, 42)
}, scoring="neg_mean_squared_error")

model.fit(X_train, y_train)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)

train_mae = mean_absolute_error(y_train, y_hat_train)
test_mae = mean_absolute_error(y_test, y_hat_test)

print(train_mse)
print(test_mse)
print(train_mae)
print(test_mae)