import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("properties_info_updated.csv")

df = df[~df['Floor'].isin(['1', '4'])]

# Replace 'Ground' with '0' and transform basement values
df['Floor'] = df['Floor'].str.replace('Ground', '0')
df['Floor'] = df['Floor'].str.replace('Upper Basement', '-1')
df['Floor'] = df['Floor'].str.replace('Lower Basement', '-2')

# Assign '0 out of 1' to single-family residences with missing Floor values
df['Floor'] = df.apply(lambda row: '0 out of 1' if row['Property Type'] == 'SingleFamilyResidence' and pd.isnull(row['Floor']) else row['Floor'], axis=1)

# Extract floor number and total floors
df[['Floor Number', 'Total Floors']] = df['Floor'].str.extract(r'(-?\d+) out of (\d+)')

# Convert extracted values to numeric
df['Floor Number'] = pd.to_numeric(df['Floor Number'])
df['Total Floors'] = pd.to_numeric(df['Total Floors'])

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

# Clean and convert 'Area' column to numeric
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

# Drop irrelevant columns
df = df.drop(['Address Locality', 'Address Country', 'Currency'], axis=1)

# Dummy for missing floors (for interaction variables)
df['Floor_present'] = df['Floor'].notna().astype(int)

# Save the cleaned dataset
df.to_csv("cleaned_properties_final.csv", index=False)