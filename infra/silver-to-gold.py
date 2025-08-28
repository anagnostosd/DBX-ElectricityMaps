from pyspark.sql.functions import col, hour, dayofweek, month, lag, avg, window, lit, to_timestamp, to_utc_timestamp, abs, round, unix_timestamp, datediff, date_format, to_date, dayofyear, weekofyear
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

# We'll use the most recent ingested_at timestamp to define our 24h historic window
latest_ingestion_time = spark.read.format("delta").table("silver.energy_data").select(F.max("ingested_at")).collect()[0][0]
historic_data_start = latest_ingestion_time - datetime.timedelta(hours=24)

# 2. Read and filter data for the historic window
silver_energy_df = spark.read.format("delta").table("silver.energy_data") \
    .filter(col("datetime") >= historic_data_start) \
    .filter(col("ingested_at") == latest_ingestion_time)

# We will read only the most recent data from the silver weather table based on ingestion time
silver_weather_df = spark.read.format("delta").table("silver.weather_forecast_data") \
    .filter(col("ingested_at") == latest_ingestion_time)

# 3. Split weather data into historic observations and future forecasts based on ingestion time
# Historic weather data: where forecast datetime is in the past, relative to the ingestion time
weather_historic_df = silver_weather_df.filter(col("datetime") <= latest_ingestion_time)

# Forecast data: where the forecast datetime is in the future, relative to the ingestion time
weather_forecast_df = silver_weather_df.filter(col("datetime") > latest_ingestion_time)


# 4. Pivot the historic weather data from long to wide format
weather_cols = ["temp", "humidity", "precip", "windspeed", "solarenergy", "winddir"]
locations = ["Athens, Greece", "Heraklion, Greece", "Thessaloniki, Greece"]

# The pivot function with special characters in column names can be tricky.
# We will create a list of select expressions to safely rename columns.
pivot_df = weather_historic_df.groupBy("datetime").pivot("name", locations).agg(*[F.first(col).alias(col) for col in weather_cols])

# Dynamically create the list of renamed columns
select_cols = [col("datetime")]
for location in locations:
    sanitized_location = location.replace(", Greece", "").replace(" ", "_").lower()
    for wc in weather_cols:
        select_cols.append(col(f"`{location}_{wc}`").alias(f"{wc}_{sanitized_location}"))

pivoted_historic_weather_df = pivot_df.select(select_cols)

# 5. Join the energy and pivoted historic weather dataframes to create the final historic gold table
historic_gold_df = silver_energy_df.join(pivoted_historic_weather_df, on=["datetime"], how="left_outer")

# Add more time-based features to the historic data
historic_gold_df = historic_gold_df.withColumn("hour_of_day", hour("datetime")) \
                 .withColumn("day_of_week", dayofweek("datetime")) \
                 .withColumn("month", month("datetime")) \
                 .withColumn("day_of_year", dayofyear("datetime")) \
                 .withColumn("week_of_year", weekofyear("datetime"))

# 6. Create features for the historic gold table
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

for column in lag_columns:
    historic_gold_df = historic_gold_df.withColumn(f"lag_{column}_1h", lag(col(column), 1).over(window_spec))
    historic_gold_df = historic_gold_df.withColumn(f"rolling_avg_{column}_24h", avg(col(column)).over(window_spec.rowsBetween(-23, 0)))

historic_gold_df = historic_gold_df.fillna(0)

# 7. Create a composite key and additional time features for the weather forecast table
forecast_gold_df = weather_forecast_df \
    .withColumn("ingested_hour_utc", to_timestamp(date_format(col("ingested_at"), "yyyy-MM-dd HH:00:00"))) \
    .withColumn("forecast_offset_h", (unix_timestamp(col("datetime")) - unix_timestamp(col("ingested_at"))) / 3600) \
    .withColumn("hour_of_day", hour("datetime")) \
    .withColumn("day_of_week", dayofweek("datetime")) \
    .withColumn("month", month("datetime")) \
    .withColumn("day_of_year", dayofyear("datetime")) \
    .withColumn("week_of_year", weekofyear("datetime")) \
    .select(
        "ingested_hour_utc",
        "datetime",
        "name",
        "forecast_offset_h",
        "hour_of_day",
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        "temp", "humidity", "precip", "windspeed", "winddir", "cloudcover", "solarenergy"
    )

# 8. Write the final DataFrames to the gold layer tables
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")

# Write to historic data table
historic_table_name = "gold.historic_data"
if not spark.catalog.tableExists(historic_table_name):
    historic_gold_df.write.format("delta").mode("overwrite").saveAsTable(historic_table_name)
else:
    delta_table = DeltaTable.forName(spark, historic_table_name)
    delta_table.alias("old_data") \
        .merge(
            historic_gold_df.alias("new_data"),
            "old_data.zone = new_data.zone AND old_data.datetime = new_data.datetime"
        ) \
        .whenMatchedUpdateAll() \
        .whenNotMatchedInsertAll() \
        .execute()
print("Gold layer historic data update complete.")

# Write to weather forecasts table
forecast_table_name = "gold.weather_forecasts"
if not spark.catalog.tableExists(forecast_table_name):
    forecast_gold_df.write.format("delta").mode("overwrite").saveAsTable(forecast_table_name)
else:
    delta_table = DeltaTable.forName(spark, forecast_table_name)
    delta_table.alias("old_data") \
        .merge(
            forecast_gold_df.alias("new_data"),
            "old_data.ingested_hour_utc = new_data.ingested_hour_utc AND old_data.datetime = new_data.datetime AND old_data.name = new_data.name"
        ) \
        .whenNotMatchedInsertAll() \
        .execute()
print("Gold layer weather forecasts update complete.")
