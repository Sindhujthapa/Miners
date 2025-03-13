import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("cleaned_properties_info_new.csv")  

df = df[df['Floor_present'] == 1]


numeric_features = ['Bathroom', 'Balcony', 'Latitude', 'Longitude', 'Area']

#dummy_features = ['Furnished', 'Semi-Furnished', 'Unfurnished', 'Bachelors', 'Bachelors/Family', 'Family', 'Unknown_Tenant Preferred', 'East', 'North', 
 #                 'North - East', 'North - West', 'South', 'South - East', 'South -West', 
  #                'Unknown_Facing', 'West', 'Garden/Park', 'Main Road', 'Pool', 'Apartment', 'SingleFamilyResidence', 
   #               'Unknown_Property Type', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 
    #              'Mumbai', 'New Delhi']

# Step 3: Drop rows where 'Bathroom' or 'Balcony' contain non-numeric values
df = df[df['Bathroom'].astype(str).str.isnumeric() & df['Balcony'].astype(str).str.isnumeric()]

# Step 4: Convert numeric features to proper data types
df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')

# Step 5: Drop rows with NaN in numeric features (to ensure clean dataset)
df = df.dropna(subset=numeric_features)

# Step 6: Define Features (X) and Target Variable (y)
features = numeric_features + dummy_features
X = df[features]
y = df['Price']

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Evaluate Model Performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R²: {train_score:.4f}")
print(f"Testing R²: {test_score:.4f}")

train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))
print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
