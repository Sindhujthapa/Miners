import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("properties_info.csv")

df = df[~df['Floor'].isin(['1', '4'])] 

# Clean if there is too little information
columns_to_check = ['Bathroom', 'Facing', 'Floor', 'Overlooking', 'Balcony', 'Ownership']

# Drop rows where all the specified columns have missing values
df = df.dropna(subset=columns_to_check, how='all')

# Dummy variables
df_dummies = df.drop(["Rental Start Time", "Rental End Time", 'Property ID', 'Bathroom', 'Area', 'Floor', 'Balcony', 'Latitude', 'Longitude', 'Address Locality', 'Address Country', 'Price', 'Currency'], axis = 1)

for col in df_dummies.columns:
    df[col] = df[col].fillna(f'Unknown_{col}')
    df[col] = df[col].str.split(', ')
    df_exploded = df[col].explode()
    df_encoded = pd.get_dummies(df_exploded)
    df_encoded = df_encoded.groupby(df_exploded.index).sum()
    df = pd.concat([df, df_encoded], axis=1)


# Replace 'Ground' with '0'
df['Floor'] = df['Floor'].str.replace('Ground', '0')
df['Floor'] = df['Floor'].str.replace('Upper Basement', '-1')
df['Floor'] = df['Floor'].str.replace('Lower Basement', '-2')
df['Floor'] = df.apply(lambda row: '0 out of 1' if row['Property Type'] == 'SingleFamilyResidence' and pd.isnull(row['Floor']) else row['Floor'], axis=1)

# Extract floor number and total floors (e.g., '3 out of 10')
df[['Floor Number', 'Total Floors']] = df['Floor'].str.extract(r'(-?\d+) out of (\d+)')

# Convert extracted values to numeric
# df['Floor Number'] = pd.to_numeric(df['Floor Number'], errors='coerce')
# df['Total Floors'] = pd.to_numeric(df['Total Floors'], errors='coerce')

df['Floor Number'] = pd.to_numeric(df['Floor Number'])
df['Total Floors'] = pd.to_numeric(df['Total Floors'])

# Drop the original 'Floor' column if no longer needed
#df.drop(columns=['Floor'], inplace=True)

# Convert the rental date columns to datetime format
df["Rental Start Time"] = pd.to_datetime(df["Rental Start Time"], errors='coerce')
df["Rental End Time"] = pd.to_datetime(df["Rental End Time"], errors='coerce')

# Extract year, month, and day into separate columns
df["Rental Start Year"] = df["Rental Start Time"].dt.year
df["Rental Start Month"] = df["Rental Start Time"].dt.month
df["Rental Start Day"] = df["Rental Start Time"].dt.day

df["Rental End Year"] = df["Rental End Time"].dt.year
df["Rental End Month"] = df["Rental End Time"].dt.month
df["Rental End Day"] = df["Rental End Time"].dt.day

# Drop the original datetime columns if they are no longer needed
df.drop(columns=["Rental Start Time", "Rental End Time"], inplace=True)

# # Clean and convert 'Area' column to numeric
# df['Area'] = df['Area'].str.replace(' sqft', '', regex=True)  # Remove 'sqft' text
# df['Area'] = pd.to_numeric(df['Area'], errors='coerce')  # Convert to numeric

df['Area'] = df['Area'].str.replace(',', '', regex=True)

# Extract numeric values and convert to float
df['Area Value'] = df['Area'].str.extract(r'(\d+\.?\d*)').astype(float)

# Identify the unit of measurement
df['Area Unit'] = df['Area'].str.extract(r'([a-zA-Z]+)')

# Define conversion factors
conversion_factors = {
    'acre': 43560,
    'sqyrd': 9,
    'sqm': 10.7639,
    'sqft': 1  # No conversion needed
}

# Convert all areas to sqft
df['Area in sqft'] = df.apply(lambda row: row['Area Value'] * conversion_factors.get(row['Area Unit'], np.nan), axis=1)

# Drop unnecessary columns
df.drop(columns=['Area', 'Area Value', 'Area Unit'], inplace=True)

# Rename the standardized column
df.rename(columns={'Area in sqft': 'Area'}, inplace=True)

# Remove rows with missing 'Price' values
df = df.dropna(subset=['Price'])

# Fill missing values in 'Balcony' with 0.
df['Balcony'] = df['Balcony'].fillna(0)
df['Balcony'] = df['Balcony'].astype('category')

# Fill missing values in 'Bathroom' with 0.
df['Bathroom'] = df['Bathroom'].fillna(0)
df['Bathroom'] = df['Bathroom'].astype('category')

df = df.drop(['Address Locality', 'Address Country', 'Currency'], axis=1)

#Dummy for missing floors (for interaction variables)
df['Floor_present'] = df['Floor'].notna().astype(int)

# Display the cleaned dataframe
print(df.head())
print(df.info())


# Save the cleaned dataset
df.to_csv("cleaned_properties_info.csv", index=False)