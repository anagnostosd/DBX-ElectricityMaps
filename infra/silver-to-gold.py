from pyspark.sql.functions import col, hour, dayofweek, month, lag, avg, window, lit, to_timestamp, to_utc_timestamp, row_number
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import pyspark.sql.functions as F
import datetime

# 1. Get the lookback interval from a notebook widget. Default to 48 hours.
try:
    lookback_hours = int(dbutils.widgets.get("lookback_hours"))
except:
    lookback_hours = 48
    print(f"No lookback_hours parameter provided, defaulting to {lookback_hours} hours.")

# 2. Read data from the silver layer, focusing on the most recent data
silver_energy_df = spark.read.format("delta").table("silver.energy_data") \
    .filter(f"datetime >= now() - interval {lookback_hours} hours")
    
silver_weather_df = spark.read.format("delta").table("silver.weather_forecast_data") \
    .filter(f"datetime >= now() - interval {lookback_hours} hours")

# 3. Deduplicate the weather data, keeping only the latest forecast
window_spec_weather = Window.partitionBy("name", "datetime").orderBy(col("ingested_at").desc())
silver_weather_deduped_df = silver_weather_df.withColumn("row_num", row_number().over(window_spec_weather)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# 4. Pivot the weather data from long to wide format
# Let's select the weather columns we want to pivot
weather_cols = ["temp", "humidity", "precip", "windspeed", "solarenergy", "winddir"]
pivot_df = silver_weather_deduped_df.groupBy("datetime").pivot("name").agg(*[F.first(col).alias(col) for col in weather_cols])

# Rename the pivoted columns for clarity
pivoted_weather_df = pivot_df.select(
    "datetime",
    col("Athens, Greece_temp").alias("temp_athens"),
    col("Athens, Greece_humidity").alias("humidity_athens"),
    col("Athens, Greece_precip").alias("precip_athens"),
    col("Athens, Greece_windspeed").alias("windspeed_athens"),
    col("Athens, Greece_solarenergy").alias("solarenergy_athens"),
    col("Athens, Greece_winddir").alias("winddir_athens"),
    col("Heraklion, Greece_temp").alias("temp_heraklion"),
    col("Heraklion, Greece_humidity").alias("humidity_heraklion"),
    col("Heraklion, Greece_precip").alias("precip_heraklion"),
    col("Heraklion, Greece_windspeed").alias("windspeed_heraklion"),
    col("Heraklion, Greece_solarenergy").alias("solarenergy_heraklion"),
    col("Heraklion, Greece_winddir").alias("winddir_heraklion"),
    col("Thessaloniki, Greece_temp").alias("temp_thessaloniki"),
    col("Thessaloniki, Greece_humidity").alias("humidity_thessaloniki"),
    col("Thessaloniki, Greece_precip").alias("precip_thessaloniki"),
    col("Thessaloniki, Greece_windspeed").alias("windspeed_thessaloniki"),
    col("Thessaloniki, Greece_solarenergy").alias("solarenergy_thessaloniki"),
    col("Thessaloniki, Greece_winddir").alias("winddir_thessaloniki")
)

# 5. Join the energy and pivoted weather dataframes
gold_df = silver_energy_df.join(pivoted_weather_df, on=["datetime"], how="left_outer")

# 6. Handle potential missing values by filling with a reasonable default
# In this case, we'll fill nulls with 0. In a real-world scenario, you might use
# more advanced imputation techniques.
gold_df = gold_df.fillna(0)

# 7. Create time-based features
gold_df = gold_df.withColumn("hour", hour("datetime")) \
                 .withColumn("day_of_week", dayofweek("datetime")) \
                 .withColumn("month", month("datetime"))

# 8. Create time-series features (lagged values and rolling averages)
# We need to use a window function to perform these calculations correctly
window_spec = Window.partitionBy("zone").orderBy("datetime")

# Define the columns for which we want to create lagged features and rolling averages
lag_columns = [
    "consumption_nuclear", "consumption_geothermal", "consumption_biomass",
    "consumption_coal", "consumption_wind", "consumption_solar", "consumption_hydro",
    "consumption_gas", "consumption_oil", "consumption_total",
    "production_nuclear", "production_geothermal", "production_biomass",
    "production_coal", "production_wind", "production_solar", "production_hydro",
    "production_gas", "production_oil", "production_total",
    "import_total", "export_total", "carbon_intensity",
    "import_AL", "import_BG", "import_MK", "import_TR",
    "export_AL", "export_BG", "export_MK", "export_TR",
    "temp_athens", "humidity_athens", "precip_athens", "windspeed_athens", "solarenergy_athens", "winddir_athens",
    "temp_heraklion", "humidity_heraklion", "precip_heraklion", "windspeed_heraklion", "solarenergy_heraklion", "winddir_heraklion",
    "temp_thessaloniki", "humidity_thessaloniki", "precip_thessaloniki", "windspeed_thessaloniki", "solarenergy_thessaloniki", "winddir_thessaloniki"
]

# Add lagged 1-hour and a 24-hour rolling average for all relevant columns
for column in lag_columns:
    gold_df = gold_df.withColumn(f"lag_{column}_1h", lag(col(column), 1).over(window_spec))
    gold_df = gold_df.withColumn(f"rolling_avg_{column}_24h", avg(col(column)).over(window_spec.rowsBetween(-23, 0)))

# 9. Handle nulls created by the window functions (at the start of the time series)
gold_df = gold_df.fillna(0)

# 10. Write the final DataFrame to the gold layer using an efficient merge strategy
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
gold_table_name = "gold.ml_features"

if not spark.catalog.tableExists(gold_table_name):
    # If the table doesn't exist, create it with a simple write
    print("Gold table does not exist. Creating it...")
    gold_df.write.format("delta").mode("overwrite").saveAsTable(gold_table_name)
else:
    # If the table exists, perform a MERGE
    print("Gold table exists. Merging new data...")
    delta_table = DeltaTable.forName(spark, gold_table_name)
    
    delta_table.alias("old_data") \
        .merge(
            gold_df.alias("new_data"),
            "old_data.zone = new_data.zone AND old_data.datetime = new_data.datetime"
        ) \
        .whenMatchedUpdateAll() \
        .whenNotMatchedInsertAll() \
        .execute()
