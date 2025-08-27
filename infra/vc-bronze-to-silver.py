from pyspark.sql.functions import col, from_unixtime, unix_timestamp, to_timestamp, row_number, lit, to_utc_timestamp
from pyspark.sql.window import Window
from delta.tables import DeltaTable
from io import StringIO
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType

# 1. Get the lookback interval from a notebook widget. Default to 48 hours.
try:
    lookback_hours = int(dbutils.widgets.get("lookback_hours"))
except:
    lookback_hours = 48
    print(f"No lookback_hours parameter provided, defaulting to {lookback_hours} hours.")

# Get the timezone from a notebook widget. Default to EET.
try:
    timezone = dbutils.widgets.get("timezone")
except:
    timezone = "EET"
    print(f"No timezone parameter provided, defaulting to {timezone}.")


# 2. Read the raw data from the bronze layer, focusing on the most recent data
# We'll read data ingested within the specified lookback period to capture all recent changes.
bronze_df = spark.read.format("delta").table("bronze.weather_forecast_data") \
    .filter(f"ingested_at >= now() - interval {lookback_hours} hours")

# 3. Define the final schema for the weather forecast data
# Added the 'ingested_at' to the schema
weather_schema = StructType([
    StructField("name", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("datetime", TimestampType(), True),
    StructField("temp", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("precip", DoubleType(), True),
    StructField("windspeed", DoubleType(), True),
    StructField("winddir", DoubleType(), True),
    StructField("cloudcover", DoubleType(), True),
    StructField("solarenergy", DoubleType(), True),
    StructField("stations", StringType(), True),
    StructField("ingested_at", TimestampType(), True)
])

# 4. Function to parse and flatten the CSV data
def parse_and_flatten_csv(row):
    """
    Parses the raw CSV string into a DataFrame row by row.
    This function is necessary because Visual Crossing returns a CSV, not a structured JSON.
    """
    raw_csv = row['raw_csv']
    # Use pandas to read the CSV, skipping the first row which is metadata
    csv_data = StringIO(raw_csv)
    df = pd.read_csv(csv_data, skiprows=1)
    
    # Rename columns to match the defined schema
    df.columns = ["name", "latitude", "longitude", "datetime", "temp", "humidity", "precip", "windspeed", "winddir", "cloudcover", "solarenergy", "stations"]
    
    # Add the ingestion timestamp to every row
    df['ingested_at'] = row['ingested_at']
    
    # Convert pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    
    # Convert the 'datetime' column to string, then to timestamp, then to UTC
    spark_df = spark_df.withColumn(
        "datetime",
        to_utc_timestamp(
            to_timestamp(col("datetime").cast("string")), timezone
        )
    )
    
    return spark_df

# 5. Process and union all locations
final_silver_df = None
for row in bronze_df.collect():
    location_df = parse_and_flatten_csv(row)
    if final_silver_df is None:
        final_silver_df = location_df
    else:
        final_silver_df = final_silver_df.unionByName(location_df)
        
# 6. Perform a MERGE to incrementally update the silver table
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
silver_table_name = "silver.weather_forecast_data"

if not spark.catalog.tableExists(silver_table_name):
    # If the table doesn't exist, create it with a simple write
    print("Silver weather table does not exist. Creating it...")
    final_silver_df.write.format("delta").mode("overwrite").saveAsTable(silver_table_name)
else:
    # If the table exists, perform a MERGE on the combined dataframe
    print("Silver weather table exists. Merging new data...")
    delta_table = DeltaTable.forName(spark, silver_table_name)
    
    # The merge condition now includes 'ingested_at' to handle unique forecasts
    delta_table.alias("old_data") \
        .merge(
            final_silver_df.alias("new_data"),
            "old_data.name = new_data.name AND old_data.datetime = new_data.datetime AND old_data.ingested_at = new_data.ingested_at"
        ) \
        .whenNotMatchedInsertAll() \
        .execute()
