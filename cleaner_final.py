import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("properties_info_updated.csv")
df = df[~df['Floor'].isin(['1', '4'])]

df_dummies = df['Overlooking'].str.split(', ')
df_exploded = df_dummies.explode()
df_encoded = pd.get_dummies(df_exploded)
df_encoded = df_encoded.groupby(df_exploded.index).sum()
df = pd.concat([df, df_encoded], axis=1)  # Merge with original dataframe

# Replace 'Ground' with '0' and transform basement values
df['Floor'] = df['Floor'].str.replace('Ground', '0')
df['Floor'] = df['Floor'].str.replace('Upper Basement', '-1')
df['Floor'] = df['Floor'].str.replace('Lower Basement', '-2')

# Assign '0 out of 1' to single-family residences with missing Floor values
df['Floor'] = df.apply(lambda row: '0 out of 1' if row['Property Type'] == 'SingleFamilyResidence' and pd.isnull(row['Floor']) else row['Floor'], axis=1)

df[['Floor Number', 'Total Floors']] = df['Floor'].str.extract(r'(-?\d+) out of (\d+)')

df['Floor Number'] = pd.to_numeric(df['Floor Number'])
df['Total Floors'] = pd.to_numeric(df['Total Floors'])

df["Rental Start Time"] = pd.to_datetime(df["Rental Start Time"], errors='coerce')
df["Rental End Time"] = pd.to_datetime(df["Rental End Time"], errors='coerce')

df["Rental Start Year"] = df["Rental Start Time"].dt.year
df["Rental Start Month"] = df["Rental Start Time"].dt.month
df["Rental Start Day"] = df["Rental Start Time"].dt.day

df["Rental End Year"] = df["Rental End Time"].dt.year
df["Rental End Month"] = df["Rental End Time"].dt.month
df["Rental End Day"] = df["Rental End Time"].dt.day

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

df['Area in sqft'] = df.apply(lambda row: row['Area Value'] * conversion_factors.get(row['Area Unit'], np.nan), axis=1)

df.drop(columns=['Area', 'Area Value', 'Area Unit'], inplace=True)

df.rename(columns={'Area in sqft': 'Area'}, inplace=True)

df = df.dropna(subset=['Price'])

df = df.drop(['Address Locality', 'Address Country', 'Currency'], axis=1)

df['Floor_present'] = df['Floor'].notna().astype(int)

initial = len(df)
print('Initial Dataset length:', len(df))

plt.figure(figsize=(12, 6))
sns.boxplot(x="Address Region", y="Price", data=df)

# Add titles and labels for better readability
plt.title("Price Distribution by Address Region")
plt.xlabel("Address Region")
plt.ylabel("Price")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# Calculate the 1th and 99th percentiles of the Price column
lower_bound = df["Price"].quantile(0.01)
upper_bound = df["Price"].quantile(0.99)


df = df[(df["Price"] >= lower_bound) & (df["Price"] <= upper_bound)]

print('Final Dataset length:', len(df))
print('Dropping price outliers changed df by:', initial - len(df))


plt.figure(figsize=(12, 6))  
sns.boxplot(x="Address Region", y="Price", data=df)

plt.title("Price Distribution by Address Region")
plt.xlabel("Address Region")
plt.ylabel("Price")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

df.value_counts('Balcony')

df.value_counts('Bathroom')

df = df.loc[df['Balcony'] != '> 10']
df = df.loc[df['Bathroom'] != '> 10']

# We assume that missing values imply that there are 0 bathrooms or balconies
df["Bathroom"] = df["Bathroom"].fillna(0).astype(int)
df["Balcony"] = df["Balcony"].fillna(0).astype(int)

# Availability also has a low count of missing values. For simplicity, we drop them too
df.value_counts('Availability', dropna=False)

df = df.dropna(subset = 'Availability')
df.value_counts('Availability', dropna=False)

# Define a mapping for month abbreviations to numbers
month_mapping = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}


def compute_months_until_available(row):
    availability = row["Availability"]  # Remove quotation marks

    if availability == "Immediately":
        return 0  

    try:
        parts = availability.split()
        avail_month = month_mapping[parts[1][:3]]  # Convert "Apr" to 4, "May" to 5, etc.
        avail_year = int("20" + parts[2].strip("'"))  # Convert "25" to 2025, "26" to 2026, etc.

        # Compute months difference
        year_diff = avail_year - row["Rental Start Year"]
        month_diff = avail_month - row["Rental Start Month"]
       
        return (year_diff * 12) + month_diff
    except:
        return None  

df["Months Until Available"] = df.apply(compute_months_until_available, axis=1)

# Finally, we create some distances to the city center to account for the position in the linear/lasso regression

city_centers = {
    "Hyderabad": (17.4506, 78.3812),
    "Mumbai": (19.0688, 72.8703),
    "Kolkata": (22.55445, 88.34985),
    "New Delhi": (28.612894, 77.229446),
    "Bangalore": (12.975, 77.61),
    "Chennai": (13.0832, 80.2755),
}

df["City_Center_Latitude"] = df["Address Region"].map(lambda x: city_centers.get(x, (None, None))[0])
df["City_Center_Longitude"] = df["Address Region"].map(lambda x: city_centers.get(x, (None, None))[1])


df["Distance_to_City_Center"] = np.sqrt(
    ((df["Latitude"] - df["City_Center_Latitude"]) * 111) ** 2 +
    ((df["Longitude"] - df["City_Center_Longitude"]) * 111 * np.cos(np.radians(df["Latitude"]))) ** 2)

missing_counts = df.isna().sum()
print(missing_counts)

print(len(df))


df.to_csv("cleaned_properties_final.csv", index=False)
