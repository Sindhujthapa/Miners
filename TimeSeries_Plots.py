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


df_time = df_time[df_time["City"] == "Chennai"]



df_time['Weekday'] = df_time['Posted On'].dt.day_name()
weekday_counts = df_time.groupby(['Posted On', 'Weekday']).size().unstack(fill_value=0)


weekday_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')

plt.xlabel("Rental Start Date")
plt.ylabel("Number of Postings")
plt.title("Distribution of Rental Start Dates by Weekday")
plt.xticks(ticks=np.arange(0, len(weekday_counts), step=7), labels=weekday_counts.index[::7].strftime('%Y-%m-%d'), rotation=45)
plt.legend(title="Day of Week")

plt.show()


df_time['Weekday'] = df_time['Posted On'].dt.day_name()
weekday_counts = df_time['Weekday'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)


plt.figure(figsize=(10, 5))
plt.plot(weekday_counts.index, weekday_counts.values, marker='o', linestyle='-')

plt.xlabel("Day of the Week")
plt.ylabel("Total Number of Postings")
plt.title("Total Rental Postings by Day of the Week")
plt.grid(True)

plt.show()

df_time['Year-Week'] = df_time['Posted On'].dt.strftime('%Y-%U') 
df_time['Weekday'] = df_time['Posted On'].dt.day_name()


heatmap_data = df_time.groupby(['Year-Week', 'Weekday']).size().unstack().fillna(0)


heatmap_data = heatmap_data[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]


plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data.T, cmap="coolwarm", linewidths=0.5, annot=False)

plt.xlabel("Week of the Year")
plt.ylabel("Day of the Week")
plt.title("Rental Postings Heatmap (Weekday vs. Week)")
plt.show()

# Analysis including the average price
df_time['Posted On'] = pd.to_datetime(df_time['Posted On'])

df_agg = df_time.groupby('Posted On').agg(
    Num_Postings=('Posted On', 'count'),
    Avg_Price=('Rent', 'mean')).reset_index()

df_agg = df_agg[df_agg['Posted On'] != df_agg['Posted On'].min()]

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(df_agg['Posted On'], df_agg['Num_Postings'], color='skyblue', alpha=0.7, label="Number of Postings")

ax2 = ax1.twinx()
ax2.plot(df_agg['Posted On'], df_agg['Avg_Price'], color='red', marker='o', linestyle='-', label="Avg Listing Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Postings", color='blue')
ax2.set_ylabel("Average Listing Price", color='red')
plt.title("Daily Rental Postings & Average Listing Price")
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7)) 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()

df_time['Weekday'] = df_time['Posted On'].dt.day_name()

df_time['Log_Price'] = np.log1p(df_time['Rent'])

df_time_cleaned = df_time[~((df_time['Weekday'] == 'Wednesday') & (df_time['Rent'] > 3000000))]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_time_cleaned, x='Weekday', y='Log_Price', order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.xlabel("Day of the Week")
plt.ylabel("Listing Price")
plt.title("Distribution of Rental Prices by Day of the Week (Outlier Removed)")
plt.xticks(rotation=45)
plt.grid(True)

plt.show()

df_time['Year-Week'] = df_time['Posted On'].dt.strftime('%Y-%U')
heatmap_data = df_time.groupby(['Year-Week', 'Weekday'])['Rent'].mean().unstack().fillna(0)
heatmap_data = heatmap_data[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data.T, cmap="coolwarm", linewidths=0.5, annot=False)

plt.xlabel("Week of the Year")
plt.ylabel("Day of the Week")
plt.title("Heatmap of Rental Prices by Weekday Over Time")
plt.show()
