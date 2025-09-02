import datetime
from pyspark.sql.functions import col, to_timestamp, date_format, unix_timestamp, count, collect_list, struct, when, element_at, flatten, explode, lit, hour, dayofweek, month, dayofyear, weekofyear
from pyspark.sql.types import DoubleType, IntegerType, StringType
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
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
    if c.startswith("lag_") or c.startswith("rolling_avg_")
]

forecast_feature_cols = [
    c for c in forecasts_df.columns
    if c.startswith(("temp_", "humidity_", "precip_", "windspeed_", "solarenergy_", "winddir_", "cloudcover_"))
]

target_column = "carbon_intensity"

# 3. Create a single, flattened training dataframe for all ingestion times
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

# 4. Union all individual training examples into a single DataFrame
if all_training_examples:
    training_data_df = all_training_examples[0]
    for df in all_training_examples[1:]:
        training_data_df = training_data_df.unionByName(df)

    # 5. Perform a chronological train-test split
    split_point = training_data_df.agg(F.percentile_approx("ingested_hour_utc", 0.8)).collect()[0][0]
    
    train_df = training_data_df.filter(col("ingested_hour_utc") <= split_point)
    test_df = training_data_df.filter(col("ingested_hour_utc") > split_point)
    
    # 6. Define the final feature and target columns for the Scikit-learn model
    historic_features_used = [c for c in historic_feature_cols if c in training_data_df.columns]
    forecast_features_used = [c for c in forecast_feature_cols if c in training_data_df.columns]
    
    all_features = historic_features_used + forecast_features_used + ["forecast_offset_h", "hour_of_day", "day_of_week", "month", "day_of_year", "week_of_year"]
    
    # Filter for only numeric columns to avoid the VectorAssembler error
    numeric_columns = [
        c for c in all_features
        if c in training_data_df.columns and training_data_df.schema[c].dataType.typeName() in ["double", "long", "integer"]
    ]

    # Convert Spark DataFrames to Pandas for Scikit-learn
    train_pd_df = train_df.toPandas()
    test_pd_df = test_df.toPandas()
    
    X_train = train_pd_df[numeric_columns]
    y_train = train_pd_df["target_val"]
    
    X_test = test_pd_df[numeric_columns]
    y_test = test_pd_df["target_val"]

    with mlflow.start_run() as run:
        # Create and train the Scikit-learn model
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        print(f"Training a Scikit-learn GradientBoostingRegressor model on {len(X_train)} rows...")
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        # Log metrics and model
        mlflow.log_param("model_type", "scikit-learn_GradientBoostingRegressor")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log the Scikit-learn model artifact
        mlflow.sklearn.log_model(
            model,
            "carbon-intensity-model",
            registered_model_name="carbon_intensity_forecaster"
        )
        
        print(f"\nTraining complete. Model evaluated on test set.")
        print(f"Test RMSE: {rmse}")
        print(f"Test R-squared: {r2}")
        print(f"MLflow Run ID: {run.info.run_id}")

else:
    print("Not enough data to create training examples.")
    training_data_df = spark.createDataFrame([], "string")
