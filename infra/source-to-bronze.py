import requests
import json
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import StructType, StringType, ArrayType, MapType, LongType, IntegerType, DoubleType, BooleanType

# 1. Retrieve the API token from the secret scope
# IMPORTANT: This notebook assumes you have already created a secret scope
# and added your API token to it using the Databricks CLI.
# Command to create the scope: databricks secrets create-scope APIs
# Command to add the secret: databricks secrets put-secret APIs electricitymaps-token
try:
    api_token = dbutils.secrets.get(scope="APIs", key="electricitymaps-token")
except Exception as e:
    print(f"Failed to get secret: {e}")

# Define the base API URL and headers
base_url = "https://api.electricitymaps.com/v3"
headers = {"auth-token": f"{api_token}"}
zone = "GR" # We will focus on Greece due to account limitations

# Define the data sources and corresponding Delta table paths
data_sources = {
    "power_data": {
        "url": f"{base_url}/power-breakdown/history?zone={zone}",
        "path": "/bronze/power_data_bronze",
        "table_name": "bronze.power_data_bronze"
    },
    "carbon_intensity": {
        "url": f"{base_url}/carbon-intensity/history?zone={zone}",
        "path": "/bronze/carbon_data_bronze",
        "table_name": "bronze.carbon_data_bronze"
    }
}

# 2. Function to fetch data and write to Delta table
def fetch_and_write_data(source_name, url, path, table_name):
    """
    Fetches data from the specified API URL and writes the raw JSON response to a Delta table.
    This function creates the bronze layer.
    """
    print(f"Fetching data from {url}...")
    try:
        # Ensure the bronze schema exists
        spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")

        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Databricks free edition has a tiny memory limit, so we are fetching small data
        # In a production environment, you would use a proper batch or streaming job
        data = response.json()
        
        # We need to wrap the JSON in a format Spark can easily write
        # We will store the full raw JSON string along with a timestamp and source
        df = spark.createDataFrame(
            [(source_name, json.dumps(data), current_timestamp())],
            ['source', 'raw_json', 'ingested_at']
        )
        
        # Write the data to a Delta table, creating it if it doesn't exist.
        # This will be our bronze layer.
        df.write.format("delta").mode("append").save(path)
        
        print(f"Successfully wrote data to Delta table at '{path}'")
        
        # Create a metastore table for easy querying
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name} USING DELTA LOCATION '{path}'")
        print(f"Successfully created or updated table '{table_name}'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 3. Main execution loop
for source_name, details in data_sources.items():
    fetch_and_write_data(source_name, details["url"], details["path"], details["table_name"])

