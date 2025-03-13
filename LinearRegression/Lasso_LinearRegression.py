import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer


df = pd.read_csv('cleaned_properties_final.csv')  
df = df[df['Floor_present'] == 1]

# Split independent variable types 
numeric_features = ['Bathroom', 'Balcony',
                    'Floor Number', 'Total Floors', 'Area', 'Distance_to_City_Center', 'Latitude', 'Longitude']

categorical_features = ['Availability', 'Facing', 'Overlooking', 'Ownership', 'Property Type', 'Tenant Preferred', 'Address Region']
features = numeric_features + categorical_features

X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scaler', StandardScaler())  
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # Impute missing values with 'unknown'
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

# Combine preprocessing for both numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  
    ('lasso', Lasso(max_iter=5000)) 
])


param_grid = {
    'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100]  # Different values of regularization strength
}


grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


train_r2_overall = best_model.score(X_train, y_train)
test_r2_overall = best_model.score(X_test, y_test)
train_mse_overall = mean_squared_error(y_train, best_model.predict(X_train))
test_mse_overall = mean_squared_error(y_test, best_model.predict(X_test))
train_mae_overall = mean_absolute_error(y_train, best_model.predict(X_train))
test_mae_overall = mean_absolute_error(y_test, best_model.predict(X_test))

results = {
    'Overall': {
        'Best Alpha': grid_search.best_params_['lasso__alpha'],
        'Training R²': train_r2_overall,
        'Testing R²': test_r2_overall,
        'Training MSE': train_mse_overall,
        'Testing MSE': test_mse_overall,
        'Training MAE': train_mae_overall,
        'Testing MAE': test_mae_overall
    }
}

# Considering each individual city
cities = df['Address Region'].unique()

for city in cities:
    print(f"\nProcessing city: {city}")
    
    city_data = df[df['Address Region'] == city]
    
    X_city = city_data[features]
    y_city = city_data['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X_city, y_city, test_size=0.2, random_state=42)
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    train_r2 = best_model.score(X_train, y_train)
    test_r2 = best_model.score(X_test, y_test)
    train_mse = mean_squared_error(y_train, best_model.predict(X_train))
    test_mse = mean_squared_error(y_test, best_model.predict(X_test))
    train_mae = mean_absolute_error(y_train, best_model.predict(X_train))
    test_mae = mean_absolute_error(y_test, best_model.predict(X_test))
    
    results[city] = {
        'Best Alpha': grid_search.best_params_['lasso__alpha'],
        'Training R²': train_r2,
        'Testing R²': test_r2,
        'Training MSE': train_mse,
        'Testing MSE': test_mse,
        'Training MAE': train_mae,
        'Testing MAE': test_mae
    }
    
    print(f"Best Alpha: {grid_search.best_params_['lasso__alpha']}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Testing MAE: {test_mae:.2f}")

# Convert results into a DataFrame for easy comparison
results_df = pd.DataFrame(results).T
print("\nResults for all cities and overall:")
print(results_df)
