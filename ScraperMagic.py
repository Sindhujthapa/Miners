import csv
import json
from bs4 import BeautifulSoup

# Read the HTML file
with open("housingchennai.html", "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")

# Create or open the CSV file for writing the results
with open("properties_info.csv", "w", newline="", encoding="utf-8") as csvfile:
    # Define the CSV writer and the header row
    fieldnames = [
        "Property ID", "Furnishing", "Bathroom", "Tenant Preferred", "Availability",
        "Area", "Floor", "Facing", "Overlooking", "Balcony", "Ownership",
        "Property Type", "Latitude", "Longitude", "Address Locality", "Address Region", "Address Country",
        "Rental Start Time", "Rental End Time", "Price", "Currency"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header to the CSV
    writer.writeheader()

    # Loop through all divs with class "mb-srp__list" to extract the property details
    for div in soup.find_all("div", class_="mb-srp__list"):
        property_id = div["id"]  # Extract the property ID
       
        # Initialize a dictionary to store the data for the current property
        property_data = {
            "Property ID": property_id,
            "Furnishing": "",
            "Bathroom": "",
            "Tenant Preferred": "",
            "Availability": "",
            "Area": "",  # Combined field for Carpet Area and Super Area
            "Floor": "",
            "Facing": "",
            "Overlooking": "",
            "Balcony": "",
            "Ownership": "",  # New Ownership field
            "Property Type": "",  # New Property Type field
            "Latitude": "",  # New Latitude field
            "Longitude": "",  # New Longitude field
            "Address Locality": "",  # New Address Locality field
            "Address Region": "",  # New Address Region field
            "Address Country": "",  # New Address Country field
            "Rental Start Time": "",  # New Rental Start Time field
            "Rental End Time": "",  # New Rental End Time field
            "Price": "",  # New Price field
            "Currency": ""  # New Currency field
        }

        # Extract relevant data for each label-value pair within this div
        for item in div.find_all("div", class_="mb-srp__card__summary__list--item"):
            # Convert both label and value to lowercase for matching
            label = item.find("div", class_="mb-srp__card__summary--label").get_text(strip=True).lower()
            value = item.find("div", class_="mb-srp__card__summary--value").get_text(strip=True)

            # Match label to appropriate field (case insensitive)
            if label == "furnishing":
                property_data["Furnishing"] = value
            elif label == "bathroom":
                property_data["Bathroom"] = value
            elif label == "tenant preferred":
                property_data["Tenant Preferred"] = value
            elif label == "availability":
                property_data["Availability"] = value
            elif label == "carpet area" or label == "super area":  # Handle both Carpet Area and Super Area
                property_data["Area"] = value
            elif label == "floor":
                property_data["Floor"] = value
            elif label == "facing":
                property_data["Facing"] = value
            elif label == "overlooking":
                property_data["Overlooking"] = value
            elif label == "balcony":
                property_data["Balcony"] = value
            elif label == "ownership":  # Handle Ownership
                property_data["Ownership"] = value

        # Extract the JSON-LD scripts (both apartment and rental details)
        scripts = div.find_all("script", type="application/ld+json")
        for script in scripts:
            # Parse the JSON data from the script
            try:
                json_data = json.loads(script.string)
                if isinstance(json_data, dict):
                    # Check if the JSON data contains a type related to Residence (Apartment, Villa, etc.)
                    if json_data.get('@type') in ['Apartment', 'SingleFamilyResidence', 'Residence']:
                        # Extract Apartment details (now handles SingleFamilyResidence as well)
                        property_data["Property Type"] = json_data.get('@type', '')
                        property_data["Latitude"] = json_data.get('geo', {}).get('latitude', '')
                        property_data["Longitude"] = json_data.get('geo', {}).get('longitude', '')
                        property_data["Address Locality"] = json_data.get('address', {}).get('addressLocality', '')
                        property_data["Address Region"] = json_data.get('address', {}).get('addressRegion', '')
                        property_data["Address Country"] = json_data.get('address', {}).get('addressCountry', '')
                    # If the JSON data is about RentAction
                    elif json_data.get('@type') == 'RentAction':
                        # Extract Rental details
                        property_data["Rental Start Time"] = json_data.get('startTime', '')
                        property_data["Rental End Time"] = json_data.get('endTime', '')
                        property_data["Price"] = json_data.get('priceSpecification', {}).get('price', '')
                        property_data["Currency"] = json_data.get('priceSpecification', {}).get('priceCurrency', '')
            except json.JSONDecodeError:
                continue  # If there's an issue with the JSON parsing, just skip

        # Write the extracted data to the CSV file
        writer.writerow(property_data)

print("Data extraction complete. Results saved in 'properties_info.csv'.")