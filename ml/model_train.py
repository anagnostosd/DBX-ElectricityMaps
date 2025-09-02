import datetime
from pyspark.sql.functions import col, to_timestamp, date_format, unix_timestamp, count, collect_list, struct, when, element_at, flatten, explode
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from delta.tables import DeltaTable
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
import os

# Set a unique experiment name for MLflow to keep runs organized
mlflow.set_registry_uri("databricks")
mlflow.set_experiment("/CarbonML")

# 1. Read the gold layer tables
historic_data_df = spark.read.format("delta").table("gold.historic_data")
forecasts_df = spark.read.format("delta").table("gold.weather_forecasts")

# 2. Get all unique ingestion times to use as keys for our training windows
ingestion_hours = (
    forecasts_df.select("ingested_hour_utc")
    .distinct()
    .sort("ingested_hour_utc")
    .collect()
)
ingestion_hours = [row[0] for row in ingestion_hours]

# Define feature and target columns
historic_feature_cols = [
    c for c in historic_data_df.columns
    if c.startswith("lag_") or c.startswith("rolling_avg_") or c in ["hour_of_day", "day_of_week", "month", "day_of_year"]
]

forecast_feature_cols = [
    "temp_athens", "humidity_athens", "precip_athens", "windspeed_athens", "solarenergy_athens", "winddir_athens", "cloudcover_athens",
    "temp_heraklion", "humidity_heraklion", "precip_heraklion", "windspeed_heraklion", "solarenergy_heraklion", "winddir_heraklion", "cloudcover_heraklion",
    "temp_thessaloniki", "humidity_thessaloniki", "precip_thessaloniki", "windspeed_thessaloniki", "solarenergy_thessaloniki", "winddir_thessaloniki", "cloudcover_thessaloniki",
    "hour_of_day", "day_of_week", "month", "day_of_year"
]
target_column = "carbon_intensity"

# 3. Create a single, flattened training dataframe for all ingestion times
all_training_examples = []
for ingestion_hour in ingestion_hours:
    # Get the last 24 hours of historic data for features and targets
    historic_data_window = historic_data_df \
        .filter(col("datetime") > (ingestion_hour - F.expr("INTERVAL 24 HOURS"))) \
        .filter(col("datetime") <= (ingestion_hour + F.expr("INTERVAL 24 HOURS"))) \
        .orderBy("datetime") \
        .na.drop()
    
    if historic_data_window.count() == 48:
        # Get the next 24 hours of weather forecasts for the prediction period
        forecast_features_window = forecasts_df \
            .filter(col("ingested_hour_utc") == ingestion_hour) \
            .filter(col("forecast_offset_h") <= 24) \
            .orderBy("datetime")

        # Get the historic features from the last 24 hours
        historic_features_window = historic_data_window \
            .filter(col("datetime") <= ingestion_hour) \
            .withColumnRenamed("datetime", "historic_datetime")
        
        # Get the carbon intensity targets from the next 24 hours
        targets_window = historic_data_window \
            .filter(col("datetime") > ingestion_hour) \
            .select(col("datetime"), col(target_column).alias("target_val"))

        if historic_features_window.count() == 24 and forecast_features_window.count() == 24 and targets_window.count() == 24:
            # Join the two feature sets
            combined_features = forecast_features_window.join(
                historic_features_window,
                (historic_features_window.historic_datetime == forecast_features_window.datetime - F.expr("INTERVAL 24 HOURS")),
                "inner"
            )

            # Join with targets
            training_example = combined_features.join(
                targets_window, F.col("datetime") == F.col("datetime"), "inner"
            )

            # Add ingestion time for training
            training_example = training_example \
                .select([col("ingested_hour_utc"), col("forecast_offset_h")] + historic_feature_cols + forecast_feature_cols + [col("target_val")])

            all_training_examples.append(training_example)


# 4. Union all individual training examples into a single DataFrame
if all_training_examples:
    training_data_df = all_training_examples[0]
    for df in all_training_examples[1:]:
        training_data_df = training_data_df.union(df)

    # 5. Perform a chronological train-test split
    # Get the split point (e.g., 80% of the data)
    split_point = training_data_df.agg(F.percentile_approx("ingestion_time", 0.8)).collect()[0][0]
    
    train_df = training_data_df.filter(col("ingestion_time") <= split_point)
    test_df = training_data_df.filter(col("ingestion_time") > split_point)
    
    # 6. Define the final feature columns for the GBT model
    final_feature_columns = historic_feature_cols + forecast_feature_cols + ["forecast_offset_h"]
    
    # Assemble the features into a single vector
    assembler = VectorAssembler(inputCols=final_feature_columns, outputCol="features")

    with mlflow.start_run() as run:
        # Create a pipeline to assemble features and train the model
        pipeline = Pipeline(stages=[assembler, GBTRegressor(featuresCol="features", labelCol="target_val", maxIter=10)])
        
        # Train the single model
        print(f"Training a single model on {train_df.count()} rows...")
        model = pipeline.fit(train_df)
        
        # Make predictions on the test set
        predictions = model.transform(test_df)
        
        # Evaluate the model
        evaluator_rmse = RegressionEvaluator(labelCol="target_val", predictionCol="prediction", metricName="rmse")
        rmse = evaluator_rmse.evaluate(predictions)
        
        evaluator_r2 = RegressionEvaluator(labelCol="target_val", predictionCol="prediction", metricName="r2")
        r2 = evaluator_r2.evaluate(predictions)
        
        # Log metrics and model
        mlflow.log_param("model_type", "GBTRegressor_unified")
        mlflow.log_param("train_size", train_df.count())
        mlflow.log_param("test_size", test_df.count())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        mlflow.spark.log_model(model, "carbon-intensity-model")
        
        print(f"\nTraining complete. Model evaluated on test set.")
        print(f"Test RMSE: {rmse}")
        print(f"Test R-squared: {r2}")
        print(f"MLflow Run ID: {run.info.run_id}")

else:
    print("Not enough data to create full 48-hour windows for training.")
    training_data_df = spark.createDataFrame([], "string")
