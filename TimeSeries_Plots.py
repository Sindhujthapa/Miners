import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates

df_cross = pd.read_csv("cleaned_properties_final.csv")
df_time = pd.read_csv("House_Rent_Dataset.csv")

df_cross.columns

df_time.columns

df_cross['posted_on'] = pd.to_datetime(
    df_cross.rename(columns={'Rental Start Year': 'year', 'Rental Start Month': 'month', 'Rental Start Day': 'day'})[['year', 'month', 'day']]
)


df_cross.head()

df_cross['posted_on'].hist(bins=30, edgecolor='black')
plt.xlabel("Rental Start Date")
plt.ylabel("Number of Postings")
plt.title("Distribution of Rental Start Dates")
plt.xticks(rotation=45)
plt.show()

df_time['Posted On'] = pd.to_datetime(df_time['Posted On'])
df_time['Posted On'].hist(bins=100, edgecolor='black')
plt.xlabel("Rental Start Date")
plt.ylabel("Number of Postings")
plt.title("Distribution of Rental Start Dates")
plt.xticks(rotation=45)
plt.show()

df_time.value_counts("City")

# Comment out to get graphs for each city
df_time = df_time[df_time["City"] == "Chennai"]
#df_time = df_time[df_time["City"] == "Hyderabad"]

# Create a pivot table to count postings per day, grouped by weekday
df_time['Weekday'] = df_time['Posted On'].dt.day_name()
weekday_counts = df_time.groupby(['Posted On', 'Weekday']).size().unstack(fill_value=0)

# Plot a stacked bar chart
weekday_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')

plt.xlabel("Rental Start Date")
plt.ylabel("Number of Postings")
plt.title("Distribution of Rental Start Dates by Weekday")
plt.xticks(ticks=np.arange(0, len(weekday_counts), step=7), labels=weekday_counts.index[::7].strftime('%Y-%m-%d'), rotation=45)
plt.legend(title="Day of Week")

plt.show()

# Aggregate by weekday
df_time['Weekday'] = df_time['Posted On'].dt.day_name()
weekday_counts = df_time['Weekday'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(weekday_counts.index, weekday_counts.values, marker='o', linestyle='-')

plt.xlabel("Day of the Week")
plt.ylabel("Total Number of Postings")
plt.title("Total Rental Postings by Day of the Week")
plt.grid(True)

plt.show()

# Extract year-week and weekday
df_time['Year-Week'] = df_time['Posted On'].dt.strftime('%Y-%U')  # Year-Week format
df_time['Weekday'] = df_time['Posted On'].dt.day_name()

# Pivot table
heatmap_data = df_time.groupby(['Year-Week', 'Weekday']).size().unstack().fillna(0)

# Reorder columns for weekday order
heatmap_data = heatmap_data[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data.T, cmap="coolwarm", linewidths=0.5, annot=False)

plt.xlabel("Week of the Year")
plt.ylabel("Day of the Week")
plt.title("Rental Postings Heatmap (Weekday vs. Week)")
plt.show()
