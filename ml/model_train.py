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
# Set environment variable to use Unity Catalog for model registration
os.environ['MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC'] = 'True'
mlflow.set_experiment("/CarbonML")

# 1. Read the gold layer tables
training_data_df = spark.read.format("delta").table("gold.training_data")

# 2. Define feature and target columns
historic_feature_cols = [
    c for c in training_data_df.columns
    if c.startswith("lag_") or c.startswith("rolling_avg_")
]

forecast_feature_cols = [
    c for c in training_data_df.columns
    if c.startswith(("temp_", "humidity_", "precip_", "windspeed_", "solarenergy_", "winddir_", "cloudcover_"))
]

target_column = "carbon_intensity"

# 3. Start training if we have data
if training_data_df:

    # 4. Perform a chronological train-test split
    split_point = training_data_df.agg(F.percentile_approx("ingested_hour_utc", 0.8)).collect()[0][0]
    
    train_df = training_data_df.filter(col("ingested_hour_utc") <= split_point)
    test_df = training_data_df.filter(col("ingested_hour_utc") > split_point)
    
    # 5. Define the final feature and target columns for the Scikit-learn model
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

        # Log the Scikit-learn model artifact with input example
        try:
            input_example = X_train.iloc[[0]]  # Use the first row as an example
            mlflow.sklearn.log_model(
            model,
            "carbon-intensity-model",
            registered_model_name="carbon_intensity_forecaster",
            input_example=input_example
            )
        except Exception as e:
            print(f"The free tier is tricky :) but the model is still logged. Error logging model: {e}")
        
        print(f"\nTraining complete. Model evaluated on test set.")
        print(f"Test RMSE: {rmse}")
        print(f"Test R-squared: {r2}")
        print(f"MLflow Run ID: {run.info.run_id}")

else:
    print("Not enough data to create training examples.")
    training_data_df = spark.createDataFrame([], "string")
