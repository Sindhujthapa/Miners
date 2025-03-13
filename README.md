# Miners

Predicting Rent Prices Using Machine Learning Models

Team Members - Nilay, Mariia, Vasco, Sindhuj
Problem Statement
Understanding rental price trends is crucial for tenants, landlords, and real estate investors. Our project aims to build predictive models that estimate rental prices based on various factors, such as location, property size, number of rooms, amenities, and prices from previous periods. We aim to provide insights into the key drivers of rental costs and improve pricing transparency in the real estate market by using different but complementary approaches.
In our first approach, we will use the features of each listed property to estimate rental prices. This approach is a 'cross-sectional' analysis of the data.
In the second approach, we use a different dataset that covers a longer time frame than the one used in the first approach. We will calculate the average daily prices for two regions and attempt to predict and forecast future rental prices. This second part is a 'time-series' analysis of the data.


## Time Series Analysis  
  
### Data Sources  
- Data [from Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset?resource=download&select=House_Rent_Dataset.csv), scraped from MagicBricks.  
- Complete dataset (no missing values). 

### Features  
- **BHK:** Number of bedrooms/halls/kitchens.  
- **Rent:** Monthly rent.  
- **Size:** Square footage.  
- **Floor:** Floor number.  
- **Area Type/Locality:** Description of location.  
- **City:** City where property is located.  
- **Furnishing Status:** Furnished, semi-furnished, or unfurnished.  
- **Tenant Preferred:** Owner’s preference.  
- **Bathroom:** Number of bathrooms.  
- **Point of Contact:** Who to contact.

The data lacks details like precise locations (latitude/longitude) and specific bedroom counts (only total BHK is available). While not critical for model estimation, including time-varying and regional variables (e.g., building age, population growth, crime rates) could have improved predictive power.

### Limitations  
1.  Chennai and Hyderabad have only 74-75 observations, limiting predictions to short-term trends.
2.  Predictions can’t cover the entire country due to regional trends and missing data for some cities.
3.  Varying observation counts across cities restrict the analysis to only the two cities with the most data.
4.  Static variables, like apartment size or bedroom count, cannot be included in the time-based analysis.
5.  Missing data on time-varying features and regional factors limits model accuracy and predictive power.
