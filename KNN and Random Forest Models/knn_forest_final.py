import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# Initialize dictionaries to store results with additional training metrics
aggregate_results = {
    'Model': [],
    'Features': [],
    'MSE': [],
    'MAE': [],
    'Train_MSE': [],
    'Train_MAE': [],
    'Optimal_K': []
}

region_results = {
    'Region': [],
    'Model': [],
    'Features': [],
    'MSE': [],
    'MAE': [],
    'Train_MSE': [],
    'Train_MAE': [],
    'Optimal_K': []
}

df = pd.read_csv("cleaned_properties_final.csv")
print(len(df))


regions = df["Address Region"].unique()


results = {}

for region in regions:
    # Filter the dataset for the current region
    df_region = df[df["Address Region"] == region]

    # Drop rows with missing Latitude or Longitude
    df_region = df_region.dropna(subset=["Latitude", "Longitude"])
   
    # Define features (latitude and longitude) and target (Price)
    X = df_region[["Latitude", "Longitude"]]
    y = df_region["Price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    param_grid = {
        'n_neighbors': list(range(1, 21))
    }

    knn = KNeighborsRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')

    # Train the model using GridSearchCV
    grid_search.fit(X_train, y_train)


    best_knn = grid_search.best_estimator_


    y_pred = best_knn.predict(X_test)


    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    

    y_train_pred = best_knn.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)


    results[region] = {
        "MAE": mae,
        "MSE": mse,
        "Train_MAE": train_mae,
        "Train_MSE": train_mse,
        "Optimal k": grid_search.best_params_["n_neighbors"]
    }
   

    region_results['Region'].append(region)
    region_results['Model'].append('KNN')
    region_results['Features'].append('Geo Only')
    region_results['MSE'].append(mse)
    region_results['MAE'].append(mae)
    region_results['Train_MSE'].append(train_mse)
    region_results['Train_MAE'].append(train_mae)
    region_results['Optimal_K'].append(grid_search.best_params_["n_neighbors"])


print("Mean Absolute Error for KNN using Latitude - Longitude")
for region, result in results.items():
    print(f"{region}: MAE = {result['MAE']}, Optimal k = {result['Optimal k']}")
    print(f"{region}: MSE = {result['MSE']}")
    print(f"{region}: Train MAE = {result['Train_MAE']}, Train MSE = {result['Train_MSE']}")


for region in regions:
    print(f"\nRunning Random Forest Regressor for Address Region: {region}")
   

    region_data = df[df["Address Region"] == region]
   

    region_data = region_data.dropna(subset=["Latitude", "Longitude"])
   

    X_region_geo = region_data[["Latitude", "Longitude"]]
    y_region = region_data["Price"]
   
   
    X_train_geo, X_test_geo, y_train_geo, y_test_geo = train_test_split(X_region_geo, y_region, random_state=42)
   
    
    rf_geo_model = RandomForestRegressor(n_estimators=100, random_state=42)
   
    
    rf_geo_model.fit(X_train_geo, y_train_geo)
   
    
    rf_geo_pred = rf_geo_model.predict(X_test_geo)
   
    
    rf_geo_mae = mean_absolute_error(y_test_geo, rf_geo_pred)
    rf_geo_mse = mean_squared_error(y_test_geo, rf_geo_pred)
   

    rf_geo_train_pred = rf_geo_model.predict(X_train_geo)
    rf_geo_train_mae = mean_absolute_error(y_train_geo, rf_geo_train_pred)
    rf_geo_train_mse = mean_squared_error(y_train_geo, rf_geo_train_pred)
   

    print(f"Random Forest MAE for {region} (Latitude and Longitude): {rf_geo_mae}")
    print(f"Random Forest MSE for {region} (Latitude and Longitude): {rf_geo_mse}")
    print(f"Random Forest Train MAE for {region} (Latitude and Longitude): {rf_geo_train_mae}")
    print(f"Random Forest Train MSE for {region} (Latitude and Longitude): {rf_geo_train_mse}")
   

    region_results['Region'].append(region)
    region_results['Model'].append('Random Forest')
    region_results['Features'].append('Geo Only')
    region_results['MSE'].append(rf_geo_mse)
    region_results['MAE'].append(rf_geo_mae)
    region_results['Train_MSE'].append(rf_geo_train_mse)
    region_results['Train_MAE'].append(rf_geo_train_mae)
    region_results['Optimal_K'].append(None)


X = df.drop(columns=["Price", "Property ID"]) 
y = df["Price"]  


geographical_cols = ["Latitude", "Longitude"]
categorical_cols = ["Furnishing", "Tenant Preferred", "Facing", "Ownership", "Property Type", "Address Region"]
numerical_cols = ["Bathroom", "Balcony", "Area", "Months Until Available", "Floor Number", "Total Floors"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),  
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore')), 
    ("scaler", StandardScaler())  
])


numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])


geographical_transformer = Pipeline(steps=[
    ("identity", "passthrough") 
])


X_cat = X[categorical_cols]

X_train, X_test, y_train, y_test = train_test_split(X_cat, y, random_state=42)


knn_cat_pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[("cat", categorical_transformer, categorical_cols)])),
    ("knn", KNeighborsRegressor())  
])


rf_cat_pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[("cat", categorical_transformer, categorical_cols)])),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])


param_grid_knn = {
    'knn__n_neighbors': list(range(1, 21))  # Trying k values from 1 to 20
}


grid_search_knn = GridSearchCV(knn_cat_pipeline, param_grid_knn, scoring='neg_mean_absolute_error', n_jobs=-1)


grid_search_knn.fit(X_train, y_train)


best_knn_cat = grid_search_knn.best_estimator_


knn_pred_cat = best_knn_cat.predict(X_test)
knn_rmse_cat = mean_squared_error(y_test, knn_pred_cat)
knn_mae_cat = mean_absolute_error(y_test, knn_pred_cat)


knn_train_pred_cat = best_knn_cat.predict(X_train)
knn_train_rmse_cat = mean_squared_error(y_train, knn_train_pred_cat)
knn_train_mae_cat = mean_absolute_error(y_train, knn_train_pred_cat)


print(f"KNN Regressor for Categorical Variables MSE: {knn_rmse_cat}")
print(f"KNN Regressor for Categorical Variables MAE: {knn_mae_cat}")
print(f"KNN Regressor for Categorical Variables Train MSE: {knn_train_rmse_cat}")
print(f"KNN Regressor for Categorical Variables Train MAE: {knn_train_mae_cat}")
print(f"Optimal n_neighbors for KNN: {grid_search_knn.best_params_['knn__n_neighbors']}")


aggregate_results['Model'].append('KNN')
aggregate_results['Features'].append('Categorical Only')
aggregate_results['MSE'].append(knn_rmse_cat)
aggregate_results['MAE'].append(knn_mae_cat)
aggregate_results['Train_MSE'].append(knn_train_rmse_cat)
aggregate_results['Train_MAE'].append(knn_train_mae_cat)
aggregate_results['Optimal_K'].append(grid_search_knn.best_params_['knn__n_neighbors'])


rf_cat_pipeline.fit(X_train, y_train)
rf_pred_cat = rf_cat_pipeline.predict(X_test)
rf_rmse_cat = mean_squared_error(y_test, rf_pred_cat)
rf_mae_cat = mean_absolute_error(y_test, rf_pred_cat)


rf_cat_train_pred = rf_cat_pipeline.predict(X_train)
rf_cat_train_rmse = mean_squared_error(y_train, rf_cat_train_pred)
rf_cat_train_mae = mean_absolute_error(y_train, rf_cat_train_pred)

print(f"Random Forest Regressor for Categorical Variables MSE: {rf_rmse_cat}")
print(f"Random Forest Regressor for Categorical Variables MAE: {rf_mae_cat}")
print(f"Random Forest Regressor for Categorical Variables Train MSE: {rf_cat_train_rmse}")
print(f"Random Forest Regressor for Categorical Variables Train MAE: {rf_cat_train_mae}")


aggregate_results['Model'].append('Random Forest')
aggregate_results['Features'].append('Categorical Only')
aggregate_results['MSE'].append(rf_rmse_cat)
aggregate_results['MAE'].append(rf_mae_cat)
aggregate_results['Train_MSE'].append(rf_cat_train_rmse)
aggregate_results['Train_MAE'].append(rf_cat_train_mae)
aggregate_results['Optimal_K'].append(None)


X_num = X[numerical_cols].copy()  
X_num = X_num.dropna()


y_num = y[X_num.index]

X_train, X_test, y_train, y_test = train_test_split(X_num, y_num, random_state=42)


knn_num_pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols)])),
    ("knn", KNeighborsRegressor())  
])


rf_num_pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols)])),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])


grid_search_knn_num = GridSearchCV(knn_num_pipeline, param_grid_knn, scoring='neg_mean_absolute_error', n_jobs=-1)


grid_search_knn_num.fit(X_train, y_train)


best_knn_num = grid_search_knn_num.best_estimator_


knn_pred_num = best_knn_num.predict(X_test)
knn_rmse_num = mean_squared_error(y_test, knn_pred_num)
knn_mae_num = mean_absolute_error(y_test, knn_pred_num)


knn_train_pred_num = best_knn_num.predict(X_train)
knn_train_rmse_num = mean_squared_error(y_train, knn_train_pred_num)
knn_train_mae_num = mean_absolute_error(y_train, knn_train_pred_num)


print(f"KNN Regressor for Numerical Variables MSE: {knn_rmse_num}")
print(f"KNN Regressor for Numerical Variables MAE: {knn_mae_num}")
print(f"KNN Regressor for Numerical Variables Train MSE: {knn_train_rmse_num}")
print(f"KNN Regressor for Numerical Variables Train MAE: {knn_train_mae_num}")
print(f"Optimal n_neighbors for KNN: {grid_search_knn_num.best_params_['knn__n_neighbors']}")


aggregate_results['Model'].append('KNN')
aggregate_results['Features'].append('Numerical Only')
aggregate_results['MSE'].append(knn_rmse_num)
aggregate_results['MAE'].append(knn_mae_num)
aggregate_results['Train_MSE'].append(knn_train_rmse_num)
aggregate_results['Train_MAE'].append(knn_train_mae_num)
aggregate_results['Optimal_K'].append(grid_search_knn_num.best_params_['knn__n_neighbors'])


rf_num_pipeline.fit(X_train, y_train)
rf_pred_num = rf_num_pipeline.predict(X_test)
rf_rmse_num = mean_squared_error(y_test, rf_pred_num)
rf_mae_num = mean_absolute_error(y_test, rf_pred_num)


rf_train_pred_num = rf_num_pipeline.predict(X_train)
rf_train_rmse_num = mean_squared_error(y_train, rf_train_pred_num)
rf_train_mae_num = mean_absolute_error(y_train, rf_train_pred_num)

print(f"Random Forest Regressor for Numerical Variables MSE: {rf_rmse_num}")
print(f"Random Forest Regressor for Numerical Variables MAE: {rf_mae_num}")
print(f"Random Forest Regressor for Numerical Variables Train MSE: {rf_train_rmse_num}")
print(f"Random Forest Regressor for Numerical Variables Train MAE: {rf_train_mae_num}")


aggregate_results['Model'].append('Random Forest')
aggregate_results['Features'].append('Numerical Only')
aggregate_results['MSE'].append(rf_rmse_num)
aggregate_results['MAE'].append(rf_mae_num)
aggregate_results['Train_MSE'].append(rf_train_rmse_num)
aggregate_results['Train_MAE'].append(rf_train_mae_num)
aggregate_results['Optimal_K'].append(None)


X_all = X.copy()
X_all = X_all.dropna(subset=numerical_cols + geographical_cols)  

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y[X_all.index], random_state=42)


knn_all_pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("geo", geographical_transformer, geographical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])),
    ("knn", KNeighborsRegressor())  
])


rf_all_pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("geo", geographical_transformer, geographical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])


grid_search_knn_all = GridSearchCV(knn_all_pipeline, param_grid_knn, scoring='neg_mean_absolute_error', n_jobs=-1)


grid_search_knn_all.fit(X_train_all, y_train_all)


best_knn_all = grid_search_knn_all.best_estimator_


knn_pred_all = best_knn_all.predict(X_test_all)
knn_rmse_all = mean_squared_error(y_test_all, knn_pred_all)
knn_mae_all = mean_absolute_error(y_test_all, knn_pred_all)


knn_train_pred_all = best_knn_all.predict(X_train_all)
knn_train_rmse_all = mean_squared_error(y_train_all, knn_train_pred_all)
knn_train_mae_all = mean_absolute_error(y_train_all, knn_train_pred_all)


print(f"KNN Regressor for All Variables MSE: {knn_rmse_all}")
print(f"KNN Regressor for All Variables MAE: {knn_mae_all}")
print(f"KNN Regressor for All Variables Train MSE: {knn_train_rmse_all}")
print(f"KNN Regressor for All Variables Train MAE: {knn_train_mae_all}")
print(f"Optimal n_neighbors for KNN: {grid_search_knn_all.best_params_['knn__n_neighbors']}")


aggregate_results['Model'].append('KNN')
aggregate_results['Features'].append('All Variables')
aggregate_results['MSE'].append(knn_rmse_all)
aggregate_results['MAE'].append(knn_mae_all)
aggregate_results['Train_MSE'].append(knn_train_rmse_all)
aggregate_results['Train_MAE'].append(knn_train_mae_all)
aggregate_results['Optimal_K'].append(grid_search_knn_all.best_params_['knn__n_neighbors'])


rf_all_pipeline.fit(X_train_all, y_train_all)
rf_pred_all = rf_all_pipeline.predict(X_test_all)
rf_rmse_all = mean_squared_error(y_test_all, rf_pred_all)
rf_mae_all = mean_absolute_error(y_test_all, rf_pred_all)


rf_train_pred_all = rf_all_pipeline.predict(X_train_all)
rf_train_rmse_all = mean_squared_error(y_train_all, rf_train_pred_all)
rf_train_mae_all = mean_absolute_error(y_train_all, rf_train_pred_all)

print(f"Random Forest Regressor for All Variables MSE: {rf_rmse_all}")
print(f"Random Forest Regressor for All Variables MAE: {rf_mae_all}")
print(f"Random Forest Regressor for All Variables Train MSE: {rf_train_rmse_all}")
print(f"Random Forest Regressor for All Variables Train MAE: {rf_train_mae_all}")


aggregate_results['Model'].append('Random Forest')
aggregate_results['Features'].append('All Variables')
aggregate_results['MSE'].append(rf_rmse_all)
aggregate_results['MAE'].append(rf_mae_all)
aggregate_results['Train_MSE'].append(rf_train_rmse_all)
aggregate_results['Train_MAE'].append(rf_train_mae_all)
aggregate_results['Optimal_K'].append(None)

categorical_cols = ["Furnishing", "Tenant Preferred", "Facing", "Ownership", "Property Type"]


regions = X["Address Region"].unique()


for region in regions:
    print(f"Processing region: {region}")
   
    
    X_region = X[X["Address Region"] == region]
    y_region = y[X_region.index]

   
    X_cat = X_region[categorical_cols]
   
    
    X_train, X_test, y_train, y_test = train_test_split(X_cat, y_region, random_state=42)

    
    knn_cat_pipeline = Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[("cat", categorical_transformer, categorical_cols)])),
        ("knn", KNeighborsRegressor())  
    ])

   
    rf_cat_pipeline = Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[("cat", categorical_transformer, categorical_cols)])),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    
    param_grid_knn = {
        'knn__n_neighbors': list(range(1, 21))  # Trying k values from 1 to 20
    }

    
    grid_search_knn = GridSearchCV(knn_cat_pipeline, param_grid_knn, scoring='neg_mean_absolute_error', n_jobs=-1)

    
    grid_search_knn.fit(X_train, y_train)

    
    best_knn_cat = grid_search_knn.best_estimator_

    
    knn_pred_cat = best_knn_cat.predict(X_test)
    knn_rmse_cat = mean_squared_error(y_test, knn_pred_cat)
    knn_mae_cat = mean_absolute_error(y_test, knn_pred_cat)

    
    knn_train_pred_cat = best_knn_cat.predict(X_train)
    knn_train_rmse_cat = mean_squared_error(y_train, knn_train_pred_cat)
    knn_train_mae_cat = mean_absolute_error(y_train, knn_train_pred_cat)

    
    print(f"KNN Regressor for Categorical Variables in {region} MSE: {knn_rmse_cat}")
    print(f"KNN Regressor for Categorical Variables in {region} MAE: {knn_mae_cat}")
    print(f"KNN Regressor for Categorical Variables in {region} Train MSE: {knn_train_rmse_cat}")
    print(f"KNN Regressor for Categorical Variables in {region} Train MAE: {knn_train_mae_cat}")
    print(f"Optimal n_neighbors for KNN in {region}: {grid_search_knn.best_params_['knn__n_neighbors']}")
   
    
    region_results['Region'].append(region)
    region_results['Model'].append('KNN')
    region_results['Features'].append('Categorical Only')
    region_results['MSE'].append(knn_rmse_cat)
    region_results['MAE'].append(knn_mae_cat)
    region_results['Train_MSE'].append(knn_train_rmse_cat)
    region_results['Train_MAE'].append(knn_train_mae_cat)
    region_results['Optimal_K'].append(grid_search_knn.best_params_['knn__n_neighbors'])

    
    rf_cat_pipeline.fit(X_train, y_train)
    rf_pred_cat = rf_cat_pipeline.predict(X_test)
    rf_rmse_cat = mean_squared_error(y_test, rf_pred_cat)
    rf_mae_cat = mean_absolute_error(y_test, rf_pred_cat)

   
    rf_train_pred_cat = rf_cat_pipeline.predict(X_train)
    rf_train_rmse_cat = mean_squared_error(y_train, rf_train_pred_cat)
    rf_train_mae_cat = mean_absolute_error(y_train, rf_train_pred_cat)

    print(f"Random Forest Regressor for Categorical Variables in {region} MSE: {rf_rmse_cat}")
    print(f"Random Forest Regressor for Categorical Variables in {region} MAE: {rf_mae_cat}")
    print(f"Random Forest Regressor for Categorical Variables in {region} Train MSE: {rf_train_rmse_cat}")
    print(f"Random Forest Regressor for Categorical Variables in {region} Train MAE: {rf_train_mae_cat}")
   
    
    region_results['Region'].append(region)
    region_results['Model'].append('Random Forest')
    region_results['Features'].append('Categorical Only')
    region_results['MSE'].append(rf_rmse_cat)
    region_results['MAE'].append(rf_mae_cat)
    region_results['Train_MSE'].append(rf_train_rmse_cat)
    region_results['Train_MAE'].append(rf_train_mae_cat)
    region_results['Optimal_K'].append(None)

    
    X_num = X_region[numerical_cols].copy()  
    X_num = X_num.dropna()

    
    y_num = y_region[X_num.index]

    X_train, X_test, y_train, y_test = train_test_split(X_num, y_num, random_state=42)

    
    knn_num_pipeline = Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols)])),
        ("knn", KNeighborsRegressor())  
    ])

    
    rf_num_pipeline = Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols)])),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    
    grid_search_knn_num = GridSearchCV(knn_num_pipeline, param_grid_knn, scoring='neg_mean_absolute_error', n_jobs=-1)

    
    grid_search_knn_num.fit(X_train, y_train)

   
    best_knn_num = grid_search_knn_num.best_estimator_

    
    knn_pred_num = best_knn_num.predict(X_test)
    knn_rmse_num = mean_squared_error(y_test, knn_pred_num)
    knn_mae_num = mean_absolute_error(y_test, knn_pred_num)

   
    knn_train_pred_num = best_knn_num.predict(X_train)
    knn_train_rmse_num = mean_squared_error(y_train, knn_train_pred_num)
    knn_train_mae_num = mean_absolute_error(y_train, knn_train_pred_num)

    
    print(f"KNN Regressor for Numerical Variables in {region} MSE: {knn_rmse_num}")
    print(f"KNN Regressor for Numerical Variables in {region} MAE: {knn_mae_num}")
    print(f"KNN Regressor for Numerical Variables in {region} Train MSE: {knn_train_rmse_num}")
    print(f"KNN Regressor for Numerical Variables in {region} Train MAE: {knn_train_mae_num}")
    print(f"Optimal n_neighbors for KNN in {region}: {grid_search_knn_num.best_params_['knn__n_neighbors']}")
   
    
    region_results['Region'].append(region)
    region_results['Model'].append('KNN')
    region_results['Features'].append('Numerical Only')
    region_results['MSE'].append(knn_rmse_num)
    region_results['MAE'].append(knn_mae_num)
    region_results['Train_MSE'].append(knn_train_rmse_num)
    region_results['Train_MAE'].append(knn_train_mae_num)
    region_results['Optimal_K'].append(grid_search_knn_num.best_params_['knn__n_neighbors'])

    
    rf_num_pipeline.fit(X_train, y_train)
    rf_pred_num = rf_num_pipeline.predict(X_test)
    rf_rmse_num = mean_squared_error(y_test, rf_pred_num)
    rf_mae_num = mean_absolute_error(y_test, rf_pred_num)

   
    rf_train_pred_num = rf_num_pipeline.predict(X_train)
    rf_train_rmse_num = mean_squared_error(y_train, rf_train_pred_num)
    rf_train_mae_num = mean_absolute_error(y_train, rf_train_pred_num)

    print(f"Random Forest Regressor for Numerical Variables in {region} MSE: {rf_rmse_num}")
    print(f"Random Forest Regressor for Numerical Variables in {region} MAE: {rf_mae_num}")
    print(f"Random Forest Regressor for Numerical Variables in {region} Train MSE: {rf_train_rmse_num}")
    print(f"Random Forest Regressor for Numerical Variables in {region} Train MAE: {rf_train_mae_num}")
   
    
    region_results['Region'].append(region)
    region_results['Model'].append('Random Forest')
    region_results['Features'].append('Numerical Only')
    region_results['MSE'].append(rf_rmse_num)
    region_results['MAE'].append(rf_mae_num)
    region_results['Train_MSE'].append(rf_train_rmse_num)
    region_results['Train_MAE'].append(rf_train_mae_num)
    region_results['Optimal_K'].append(None)

    
    X_all = X_region.copy()
    X_all = X_all.dropna(subset=numerical_cols + geographical_cols)  

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_region[X_all.index], random_state=42)

    
    knn_all_pipeline = Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("geo", geographical_transformer, geographical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])),
        ("knn", KNeighborsRegressor())  
    ])

    
    rf_all_pipeline = Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("geo", geographical_transformer, geographical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

   
    grid_search_knn_all = GridSearchCV(knn_all_pipeline, param_grid_knn, scoring='neg_mean_absolute_error', n_jobs=-1)

    
    grid_search_knn_all.fit(X_train_all, y_train_all)

   
    best_knn_all = grid_search_knn_all.best_estimator_

    
    knn_pred_all = best_knn_all.predict(X_test_all)
    knn_rmse_all = mean_squared_error(y_test_all, knn_pred_all)
    knn_mae_all = mean_absolute_error(y_test_all, knn_pred_all)

    
    knn_train_pred_all = best_knn_all.predict(X_train_all)
    knn_train_rmse_all = mean_squared_error(y_train_all, knn_train_pred_all)
    knn_train_mae_all = mean_absolute_error(y_train_all, knn_train_pred_all)

    
    print(f"KNN Regressor for All Variables in {region} MSE: {knn_rmse_all}")
    print(f"KNN Regressor for All Variables in {region} MAE: {knn_mae_all}")
    print(f"KNN Regressor for All Variables in {region} Train MSE: {knn_train_rmse_all}")
    print(f"KNN Regressor for All Variables in {region} Train MAE: {knn_train_mae_all}")
    print(f"Optimal n_neighbors for KNN in {region}: {grid_search_knn_all.best_params_['knn__n_neighbors']}")
   
    
    region_results['Region'].append(region)
    region_results['Model'].append('KNN')
    region_results['Features'].append('All Variables')
    region_results['MSE'].append(knn_rmse_all)
    region_results['MAE'].append(knn_mae_all)
    region_results['Train_MSE'].append(knn_train_rmse_all)
    region_results['Train_MAE'].append(knn_train_mae_all)
    region_results['Optimal_K'].append(grid_search_knn_all.best_params_['knn__n_neighbors'])

    
    rf_all_pipeline.fit(X_train_all, y_train_all)
    rf_pred_all = rf_all_pipeline.predict(X_test_all)
    rf_rmse_all = mean_squared_error(y_test_all, rf_pred_all)
    rf_mae_all = mean_absolute_error(y_test_all, rf_pred_all)

    
    rf_train_pred_all = rf_all_pipeline.predict(X_train_all)
    rf_train_rmse_all = mean_squared_error(y_train_all, rf_train_pred_all)
    rf_train_mae_all = mean_absolute_error(y_train_all, rf_train_pred_all)

    print(f"Random Forest Regressor for All Variables in {region} MSE: {rf_rmse_all}")
    print(f"Random Forest Regressor for All Variables in {region} MAE: {rf_mae_all}")
    print(f"Random Forest Regressor for All Variables in {region} Train MSE: {rf_train_rmse_all}")
    print(f"Random Forest Regressor for All Variables in {region} Train MAE: {rf_train_mae_all}")
   
    
    region_results['Region'].append(region)
    region_results['Model'].append('Random Forest')
    region_results['Features'].append('All Variables')
    region_results['MSE'].append(rf_rmse_all)
    region_results['MAE'].append(rf_mae_all)
    region_results['Train_MSE'].append(rf_train_rmse_all)
    region_results['Train_MAE'].append(rf_train_mae_all)
    region_results['Optimal_K'].append(None)


aggregate_df = pd.DataFrame(aggregate_results)
region_df = pd.DataFrame(region_results)


for df in [aggregate_df, region_df]:
    df['MSE'] = df['MSE'].round(2)
    df['MAE'] = df['MAE'].round(2)
    df['Train_MSE'] = df['Train_MSE'].round(2)
    df['Train_MAE'] = df['Train_MAE'].round(2)


aggregate_df.to_csv('aggregate_model_results.csv', index=False)
region_df.to_csv('region_model_results.csv', index=False)


print("\nAggregate Results Summary:")
print(aggregate_df)

print("\nRegional Results Summary:")
region_summary = region_df.groupby(['Region', 'Features', 'Model']).agg({
    'MSE': 'mean',
    'MAE': 'mean',
    'Train_MSE': 'mean',
    'Train_MAE': 'mean',
    'Optimal_K': lambda x: x.iloc[0] if pd.notna(x.iloc[0]) else None
}).reset_index()

print(region_summary)
