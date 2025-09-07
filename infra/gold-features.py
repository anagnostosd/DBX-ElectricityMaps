from pyspark.sql.functions import col, hour, dayofweek, month, lag, avg, window, lit, to_timestamp, to_utc_timestamp, abs, round, unix_timestamp, datediff, date_format, to_date, dayofyear, weekofyear, floor
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import pyspark.sql.functions as F
import datetime
import warnings

# 1. Get the lookback interval from a notebook widget. Default to 48 hours.
try:
    lookback_hours = int(dbutils.widgets.get("lookback_hours"))
except:
    lookback_hours = 12
    print(f"No lookback_hours parameter provided, defaulting to {lookback_hours} hours.")
try:
    tolerance_minutes = int(dbutils.widgets.get("tolerance_minutes"))
except:
    tolerance_minutes = 15
    print(f"No tolerance_minutes parameter provided, defaulting to {tolerance_minutes} minutes.")

# Get the latest ingestion time from the silver table to use as a dynamic anchor
try:
    latest_ingestion_time = spark.read.format("delta").table("gold.historic_data") \
        .agg(F.max(
            col("ingested_hour_utc")
        )).collect()[0][0]
except Exception as e:
    print(f"Could not get latest ingestion time from gold table. Error: {e}")

# 2. Get all unique ingestion times within the lookback window
all_ingestion_times = (spark.read.format("delta").table("gold.historic_data")
                       .select(col("ingested_hour_utc"))
                       .distinct()
                       .orderBy("ingested_hour_utc")
                       .filter(col("ingested_hour_utc") >= (latest_ingestion_time - datetime.timedelta(hours=lookback_hours)))
                       .collect())

# 3. Read the gold layer tables
historic_data_df = spark.read.format("delta").table("gold.historic_data")
forecasts_df = spark.read.format("delta").table("gold.weather_forecasts")

# 4. Get all unique ingestion times to use as keys for our training windows
ingestion_hours = [row[0] for row in all_ingestion_times]

# Define feature and target columns
historic_feature_cols = [
    c for c in historic_data_df.columns
    if c.startswith("lag_") or c.startswith("rolling_avg_")
]

forecast_feature_cols = [
    c for c in forecasts_df.columns
    if c.startswith(("temp_", "humidity_", "precip_", "windspeed_", "solarenergy_", "winddir_", "cloudcover_"))
]

target_column = "carbon_intensity"

# 5. Create a single, flattened training dataframe for all ingestion times
all_training_examples = []
for ingestion_hour in ingestion_hours:
    # Get the single row of historic data at the ingestion hour
    historic_snapshot = historic_data_df \
        .filter(col("datetime") == ingestion_hour) \
        .select(historic_feature_cols) \
        .na.drop()

    if historic_snapshot.count() == 1:
        # Get the next 24 hours of weather forecasts for the prediction period
        forecast_features_window = forecasts_df \
            .filter(col("ingested_hour_utc") == ingestion_hour) \
            .filter(col("forecast_offset_h") > 0) \
            .filter(col("forecast_offset_h") <= 24) \
            .orderBy("datetime") \
            .na.drop()

        if forecast_features_window.count() > 0:
            # Create time-based features for the forecast window
            forecast_features_window = forecast_features_window.withColumn("hour_of_day", hour("datetime")) \
                                                               .withColumn("day_of_week", dayofweek("datetime")) \
                                                               .withColumn("month", month("datetime")) \
                                                               .withColumn("day_of_year", dayofyear("datetime")) \
                                                               .withColumn("week_of_year", weekofyear("datetime"))

            # Get the carbon intensity targets from the historic data
            target_datetimes = [row[0] for row in forecast_features_window.select("datetime").collect()]
            targets_df = historic_data_df \
                .filter(col("datetime").isin(target_datetimes)) \
                .select(col("datetime"), col(target_column).alias("target_val"))
            
            if targets_df.count() == forecast_features_window.count():
                # Join the historic snapshot with the forecasts
                training_example = forecast_features_window.crossJoin(historic_snapshot)
                
                # Join with targets
                training_example = training_example.join(
                    targets_df, 
                    on="datetime", 
                    how="inner"
                )

                # Add ingestion time for training
                training_example = training_example.withColumn("ingested_hour_utc", lit(ingestion_hour))
                
                all_training_examples.append(training_example)

# 6. Union all individual training examples into a single DataFrame
if all_training_examples:
    training_data_df = all_training_examples[0]
    for df in all_training_examples[1:]:
        training_data_df = training_data_df.unionByName(df)
    
    # 7. Write the final historic gold table with incremental MERGE
    historic_table_name = "gold.training_data"
    if not spark.catalog.tableExists(historic_table_name):
        training_data_df.write.format("delta").mode("overwrite").saveAsTable(historic_table_name)
    else:
        delta_table = DeltaTable.forName(spark, historic_table_name)
        delta_table.alias("old_data") \
            .merge(
                training_data_df.alias("new_data"),
                "old.data.ingested_hour_utc = new_data.ingested_hour_utc AND old_data.datetime = new_data.datetime"
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()

print("Gold layer training data update complete.")
