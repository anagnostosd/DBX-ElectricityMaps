import datetime
import pandas as pd
from pyspark.sql.functions import col, lit, hour, dayofweek, month, dayofyear, weekofyear, date_trunc, current_timestamp
import pyspark.sql.functions as F
from delta.tables import DeltaTable
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# --- 1. Load the latest production model from the Delta table ---
model_table_name = "workspace.default.carbon_intensity_forecaster"
client=MlflowClient()
model_version = str(len(client.search_model_versions("name='workspace.default.carbon_intensity_forecaster'")))


try:
    # Read the model from the Delta table
    model_uri = "models:/" + model_table_name +"/" + model_version 
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"Successfully loaded model from '{model_table_name}' with URI '{model_uri}'")

except Exception as e:
    print(f"Could not load the model from the Delta table. Error: {e}")
    dbutils.notebook.exit("No model found in the Delta table.")


# --- 2. Prepare the data for prediction ---

# Read the latest data from the gold layer tables
historic_data_df = spark.read.format("delta").table("gold.historic_data")
forecasts_df = spark.read.format("delta").table("gold.weather_forecasts")

# Get the single row of the latest historic data to serve as our anchor
latest_ingestion_hour_utc = historic_data_df.agg(F.max(col("datetime"))).collect()[0][0]

# Get the historic snapshot for the prediction
historic_snapshot = historic_data_df \
    .filter(col("datetime") == latest_ingestion_hour_utc) \
    .limit(1)

# Get the next 24 hours of weather forecasts for the prediction period
# We find the most recent forecast for the latest ingestion hour
forecast_features_window = forecasts_df \
    .filter(col("ingested_hour_utc") == date_trunc("hour", lit(latest_ingestion_hour_utc))) \
    .filter(col("forecast_offset_h") > 0) \
    .filter(col("forecast_offset_h") <= 24) \
    .orderBy("datetime") \
    .na.drop()

# Define the feature columns that the model expects
historic_feature_cols = [c for c in historic_data_df.columns if c.startswith("lag_") or c.startswith("rolling_avg_")]
forecast_feature_cols = [c for c in forecasts_df.columns if c.startswith(("temp_", "humidity_", "precip_", "windspeed_", "solarenergy_", "winddir_", "cloudcover_"))]

# Create the time-based features for the forecast window
forecast_features_window = forecast_features_window.withColumn("hour_of_day", hour("datetime")) \
                                                   .withColumn("day_of_week", dayofweek("datetime")) \
                                                   .withColumn("month", month("datetime")) \
                                                   .withColumn("day_of_year", dayofyear("datetime")) \
                                                   .withColumn("week_of_year", weekofyear("datetime"))

time_features = ["forecast_offset_h", "hour_of_day", "day_of_week", "month", "day_of_year", "week_of_year"]
all_features = historic_feature_cols + forecast_feature_cols + time_features

# Combine the historic snapshot with the forecast features and keys in a single Spark DataFrame
full_prediction_df_spark = forecast_features_window.crossJoin(historic_snapshot.select(historic_feature_cols)) \
    .select(col("ingested_hour_utc"), col("datetime"), col("forecast_offset_h"), *[col(c) for c in all_features if c in forecast_features_window.columns or c in historic_snapshot.columns or c in time_features])

# Select only the features the model was trained on and convert to Pandas
prediction_df_pd = full_prediction_df_spark.toPandas()
prediction_features_pd = prediction_df_pd[[col for col in prediction_df_pd.columns if col in all_features]]
prediction_features_pd = prediction_features_pd.loc[:, ~prediction_features_pd.columns.duplicated()]

# --- 3. Generate Predictions ---

predictions = loaded_model.predict(prediction_features_pd)
predictions_pd_df = pd.DataFrame(predictions, columns=['prediction'])

# --- 4. Save the results to a Delta table ---
prediction_results_df_pd = pd.concat([prediction_df_pd, predictions_pd_df], axis=1)
prediction_results_df = spark.createDataFrame(prediction_results_df_pd)

# Add ingestion time and model version to the prediction results
prediction_results_df = prediction_results_df.withColumn("prediction_ingested_at", current_timestamp()) \
    .withColumn("model_version", lit(model_version)) \
    .select("ingested_hour_utc", "datetime", "prediction", "prediction_ingested_at", "model_version")

# Perform an incremental write to the predictions table
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
predictions_table_name = "gold.carbon_intensity_predictions"

if not spark.catalog.tableExists(predictions_table_name):
    prediction_results_df.write.format("delta").mode("overwrite").saveAsTable(predictions_table_name)
else:
    prediction_results_df.write.format("delta").mode("append").saveAsTable(predictions_table_name)

print(f"Successfully generated and saved 24-hour predictions to '{predictions_table_name}'.")
