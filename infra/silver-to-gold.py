from pyspark.sql.functions import col, hour, dayofweek, month, lag, avg, window, lit, to_timestamp, to_utc_timestamp, abs, round, unix_timestamp, datediff, date_format, to_date, dayofyear, weekofyear
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import pyspark.sql.functions as F
import datetime
import warnings

# 1. Get the lookback interval from a notebook widget. Default to 48 hours.
try:
    lookback_hours = int(dbutils.widgets.get("lookback_hours"))
except:
    lookback_hours = 48
    print(f"No lookback_hours parameter provided, defaulting to {lookback_hours} hours.")
try:
    tolerance_minutes = int(dbutils.widgets.get("tolerance_minutes"))
except:
    tolerance_minutes = 15
    print(f"No tolerance_minutes parameter provided, defaulting to {tolerance_minutes} minutes.")

# Get the latest ingestion time from the silver table to use as a dynamic anchor
try:
    latest_ingestion_time = spark.read.format("delta").table("silver.energy_data") \
        .agg(F.max(F.date_trunc("hour", col("ingested_at")))).collect()[0][0]
    if latest_ingestion_time is None:
        # Fallback for the very first run
        latest_ingestion_time = datetime.datetime.utcnow()
except Exception as e:
    print(f"Could not get latest ingestion time from silver table. Defaulting to now(). Error: {e}")
    latest_ingestion_time = datetime.datetime.utcnow()

# 2. Get all unique ingestion times within the lookback window
all_ingestion_times = (spark.read.format("delta").table("silver.energy_data")
                       .select(F.date_trunc("hour", col("ingested_at")).alias("ingested_hour_utc"))
                       .distinct()
                       .orderBy("ingested_hour_utc")
                       .filter(col("ingested_hour_utc") >= (latest_ingestion_time - datetime.timedelta(hours=lookback_hours)))
                       .collect())

# --- Main loop for incremental processing ---
for row in all_ingestion_times:
    ingested_hour_utc = row["ingested_hour_utc"]
    print(f"Processing ingestion_hour_utc: {ingested_hour_utc}")

    # Read and filter data for the current ingestion time
    silver_energy_df = spark.read.format("delta").table("silver.energy_data") \
        .filter(col("ingested_at") >= (ingested_hour_utc - F.expr(f"INTERVAL {tolerance_minutes} MINUTES"))) \
        .filter(col("ingested_at") <= (ingested_hour_utc + F.expr(f"INTERVAL {tolerance_minutes} MINUTES"))) \
        .filter(col("datetime") <= ingested_hour_utc)

    silver_weather_df = spark.read.format("delta").table("silver.weather_forecast_data") \
        .filter(col("ingested_at") >= (ingested_hour_utc - F.expr(f"INTERVAL {tolerance_minutes} MINUTES"))) \
        .filter(col("ingested_at") <= (ingested_hour_utc + F.expr(f"INTERVAL {tolerance_minutes} MINUTES")))

    # 3. Split weather data into historic observations and future forecasts
    weather_historic_df = silver_weather_df.filter(col("datetime") <= ingested_hour_utc)
    weather_forecast_df = silver_weather_df.filter(col("datetime") > ingested_hour_utc)
    
    # 4. Pivot the historic weather data from long to wide format
    weather_cols = ["temp", "humidity", "precip", "windspeed", "solarenergy", "winddir"]
    locations = ["Athens, Greece", "Heraklion, Greece", "Thessaloniki, Greece"]

    pivot_df = weather_historic_df.groupBy("datetime").pivot("name", locations).agg(*[F.first(col).alias(col) for col in weather_cols])
    
    select_cols = [col("datetime")]
    for location in locations:
        sanitized_location = location.replace(", Greece", "").replace(" ", "_").lower()
        for wc in weather_cols:
            select_cols.append(col(f"`{location}_{wc}`").alias(f"{wc}_{sanitized_location}"))

    pivoted_historic_weather_df = pivot_df.select(select_cols)
    
    # 5. Join the energy and pivoted historic weather dataframes
    historic_gold_df = silver_energy_df.join(pivoted_historic_weather_df, on=["datetime"], how="left_outer")

    # Add time-based features
    historic_gold_df = historic_gold_df.withColumn("hour_of_day", hour("datetime")) \
                     .withColumn("day_of_week", dayofweek("datetime")) \
                     .withColumn("month", month("datetime")) \
                     .withColumn("day_of_year", dayofyear("datetime")) \
                     .withColumn("week_of_year", weekofyear("datetime"))

    # Create lagged and rolling average features
    window_spec = Window.partitionBy("zone").orderBy("datetime")
    
    lag_columns = [
        "carbon_intensity", "powerConsumptionTotal", "powerProductionTotal", "powerImportTotal", "powerExportTotal",
        "temp_athens", "temp_thessaloniki", "temp_heraklion",
        "humidity_athens", "humidity_thessaloniki", "humidity_heraklion",
        "precip_athens", "precip_thessaloniki", "precip_heraklion",
        "windspeed_athens", "windspeed_thessaloniki", "windspeed_heraklion",
        "winddir_athens", "winddir_thessaloniki", "winddir_heraklion",
        "cloudcover_athens", "cloudcover_thessaloniki", "cloudcover_heraklion",
        "solarenergy_athens", "solarenergy_thessaloniki", "solarenergy_heraklion"
    ]

    for column in lag_columns:
        # Creating also lag_0h for easier reference to current value in models
        historic_gold_df = historic_gold_df.withColumn(f"lag_{column}_0h", lag(col(column), 0).over(window_spec))
        historic_gold_df = historic_gold_df.withColumn(f"lag_{column}_1h", lag(col(column), 1).over(window_spec))
        historic_gold_df = historic_gold_df.withColumn(f"rolling_avg_{column}_24h", avg(col(column)).over(window_spec.rowsBetween(-23, 0)))

    # Handle missing values
    historic_gold_df = historic_gold_df.fillna(0)
    
    # 6. Write the final historic gold table with incremental MERGE
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
    
    # --- Gold Weather Forecasts Table Transformation ---
    
    forecast_pivot_df = weather_forecast_df.groupBy("ingested_hour_utc", "datetime", "forecast_offset_h").pivot("name", locations).agg(*[F.first(col).alias(col) for col in weather_cols])
    
    forecast_select_cols = [col("ingested_hour_utc"), col("datetime"), col("forecast_offset_h")]
    for location in locations:
        sanitized_location = location.replace(", Greece", "").replace(" ", "_").lower()
        for wc in weather_cols:
            forecast_select_cols.append(col(f"`{location}_{wc}`").alias(f"{wc}_{sanitized_location}"))

    forecast_gold_df = forecast_pivot_df.select(forecast_select_cols)

    forecast_table_name = "gold.weather_forecasts"
    if not spark.catalog.tableExists(forecast_table_name):
        forecast_gold_df.write.format("delta").mode("overwrite").saveAsTable(forecast_table_name)
    else:
        delta_table = DeltaTable.forName(spark, forecast_table_name)
        delta_table.alias("old_data") \
            .merge(
                forecast_gold_df.alias("new_data"),
                "old_data.ingested_hour_utc = new_data.ingested_hour_utc AND old_data.datetime = new_data.datetime"
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()

print("Gold layer historic data and weather forecasts update complete.")
