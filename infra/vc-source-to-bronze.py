import requests
import json
import datetime
import pandas as pd
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import StructType, StringType, ArrayType, MapType, LongType, IntegerType, DoubleType, BooleanType

# Retrieve the API token from the secret scope
# IMPORTANT: This notebook assumes you have already created a secret scope
# and added your API token to it using the Databricks CLI.
try:
    api_token = dbutils.secrets.get(scope="APIs", key="vissualcrossing-token")
except Exception as e:
    print(f"Failed to get secret: {e}")
    api_token = "dummy-token"

# Define the base API URL and parameters
base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
params = {
    "unitGroup": "metric",
    "elements": "datetime,address,latitude,longitude,temp,humidity,precip,windspeed,winddir,cloudcover,solarenergy,stations",
    "include": "hours",
    "key": api_token,
    "contentType": "csv"
}

# Define the locations for which we want to fetch data
locations = ["Thessaloniki, Greece", "Athens, Greece", "Heraklion, Greece"]
table_name = "bronze.weather_forecast_data"

# Function to fetch data and write to Delta table
def fetch_and_write_weather_data(location, table_name):
    """
    Fetches weather forecast data for a specific location and writes the raw CSV response
    to a Delta table.
    """
    print(f"Fetching weather data for {location}...")
    try:
        url = f"{base_url}/{location}/next24hours/tomorrow"
        response = requests.request("GET", url, params=params)
        response.raise_for_status()

        # The API returns a CSV string. We'll store this as a raw string.
        raw_csv_data = response.text

        # Create a DataFrame to hold the raw CSV string and metadata
        df = spark.createDataFrame(
            [(location, raw_csv_data, datetime.datetime.now())],
            ['location_name', 'raw_csv', 'ingested_at']
        )
        
        # Ensure the bronze schema exists
        spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")

        # Write the data to a Delta table, creating it if it doesn't exist.
        df.write.format("delta").mode("append").saveAsTable(table_name)
        
        print(f"Successfully wrote data for {location} to table '{table_name}'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {location}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Main execution loop to fetch data for all locations
for location in locations:
    fetch_and_write_weather_data(location, table_name)
