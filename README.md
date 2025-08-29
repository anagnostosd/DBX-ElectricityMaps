# Electricity Maps Data Pipeline

This project demonstrates a foundational MLOps pipeline on Databricks, transforming raw data from the Electricity Maps and Visual Crossing APIs into a feature-rich, machine-learning-ready gold layer. 
The pipeline is designed to be incremental, robust, and reproducible, following the core principles of the Medallion Architecture.

## Project Overview

The primary goal of this pipeline is to ingest hourly power and carbon intensity data, as well as weather forecast data for Greece (`GR`). This data is then cleaned, enriched, and prepared for time-series forecasting. The project is structured in three key stages:

* **Bronze Layer**: Raw data ingestion.

* **Silver Layer**: Data cleaning and transformation.

* **Gold Layer**: Feature engineering for machine learning.

## How to Run This Project

To set up and run this project in your own Databricks workspace, follow these steps in order:

1.  **Get API Tokens**: Sign up for a free (or paid) Databricks account, a free Electricity Maps API token, and a free Visual Crossing API token. The free Electricity Maps API provides historical data for only the last 24 hours.

2.  **Configure Databricks Secrets**: Use the Databricks CLI on your local machine to create a secret scope and add your API tokens securely. This is a crucial step to avoid hardcoding credentials.

    * Create a secret scope named `APIs`.

    * Add your Electricity Maps token with the key `electricitymaps-token`.

    * Add your Visual Crossing token with the key `vissualcrossing-token`.

3.  **Connect to a Git Repository**: Fork this project's code and connect your Databricks workspace to the Git repository. This ensures version control and allows for collaborative development.

4.  **Deploy the Jobs**: Replay the jobs from their YAML files (or manually create them in the Databricks UI), and adjust the notebook paths as needed to match your repository structure.

This setup provides a fully automated pipeline from data ingestion to a production-ready dataset.

## Limitations

This project is designed as a foundational template, and as such, it comes with a few limitations that you should be aware of:

* **Data Availability**: The free APIs used here do not provide access to a long history of data. The Electricity Maps free tier, for example, only provides data for the last 24 hours. Consequently, your pipeline will need to run for several weeks to gather enough data to build a truly robust model.

* **Model Complexity**: The ML models chosen, such as the Gradient Boosted Tree Regressor, are deliberately simple to match the computational constraints of the free Databricks tier. The pipeline is structured to be extensible, so you can swap in more complex models (e.g., recurrent or deep networks) as needed if you upgrade your account.

* **Purpose**: This project serves as a starting point. It provides a solid, end-to-end framework for MLOps practices, allowing you to build upon it and adapt it to more complex problems as you see fit. It is not intended to be a fully optimized, production-ready solution out of the box.

## Pipeline Stages

### 1. Bronze Layer: Raw Data Ingestion

The bronze layer is the entry point for all data. Raw API responses from both the Electricity Maps and Visual Crossing APIs are fetched and stored in separate Delta tables.

* **Data Sources**: Electricity Maps API v3 and Visual Crossing Weather API.

* **Method**: A Python script periodically fetches the latest data.

* **Storage**: Data is stored in two separate Delta tables, `bronze.power_data` and `bronze.carbon_data`, for energy data. A third table, `bronze.weather_forecast_data`, stores the raw CSV data from the weather API. Each table holds the raw response, a source identifier, and a timestamp of ingestion.

* **Key Feature**: This layer is designed for idempotency. The ingestion script appends new data, and the downstream silver layer handles deduplication.

### 2. Silver Layer: Data Cleaning and Transformation

The silver layer takes the raw bronze data, flattens it, and creates a clean, structured table. This is the first step toward creating a reliable "single source of truth."

* **Inputs**:

    * Raw JSON data from `bronze.power_data` and `bronze.carbon_data`.

    * Raw CSV data from `bronze.weather_forecast_data`.

* **Transformations**:

    * **Energy Data**: The nested JSON is parsed and flattened, and power and carbon intensity data are joined on `zone` and `datetime`.

    * **Weather Data**: The raw CSV data is parsed, flattened, and time zones are aligned to UTC. Deduplication is performed to keep only the latest forecast for each datetime and location.

* **Output**: Two clean, deduplicated Delta tables: `silver.energy_data` and `silver.weather_forecast_data`.

* **Key Feature**: The process uses a Delta Lake `MERGE` statement for efficient, incremental updates, only processing new or updated data. This ensures the silver table is always up-to-date without a full rewrite.

### 3. Gold Layer: Feature Engineering

The gold layer is the final stage, where the silver data is enriched with features ready for direct use in an ML model. This layer is now split into two separate tables to better handle the distinction between historic measurements and future forecasts.

* **Gold Historic Data Table (`gold.historic_data`)**: This table contains all historic power, carbon intensity, and weather observation data. This will be the primary source for model training and backtesting. The data is deduplicated to ensure only the most recent data for each hour is included.

* **Gold Weather Forecasts Table (`gold.weather_forecasts`)**: This table stores future weather forecasts. The key for this table is a composite of the `ingested_hour_utc` and `datetime` (along with the city name), allowing us to track the forecast horizon for each unique prediction run.

* **Input**: Cleaned data from both `silver.energy_data` and `silver.weather_forecast_data` tables.

* **Transformations**:

    * **Historic Data**: Historic weather observations (where `datetime` <= `ingested_at`) are pivoted and joined with energy data.

    * **Forecasts**: Forecast data (where `datetime` > `ingested_at`) is processed separately to create a composite key and `forecast_offset` to track the prediction horizon.

* **Features Created**:

    * **Time-based features**: `hour`, `day_of_week`, `month`.

    * **Lagged features**: `lag_..._1h` (1-hour lag) for all key historic columns.

    * **Rolling averages**: `rolling_avg_..._24h` (24-hour rolling average) for the same key historic columns.

* **Output**: Two separate, comprehensive Delta tables: `gold.historic_data` and `gold.weather_forecasts`.

* **Key Feature**: This layer also uses an incremental `MERGE` strategy, making it an efficient source for continuously updated training data.

### 4. ML Training, Serving, and Monitoring

This section outlines the final stages of the MLOps pipeline, where the raw data is transformed into a functional, production-ready model.

* **Model Training**: We use a **Gradient Boosted Tree Regressor** to predict carbon intensity for the next 24 hours. The model is trained on a flattened dataset of historic features and future weather forecasts, with a chronological train-test split to prevent data leakage.

* **Experiment Tracking**: We use **MLflow** to manage and track all model training runs. It automatically logs key metrics, hyperparameters, and the model artifact itself, ensuring that all experiments are reproducible and easy to compare.

* **Model Serving (Batch Prediction) -- Ongoing**: Since real-time serving is a premium feature on Databricks, the model will be deployed for **batch prediction**. A dedicated job will read the latest data from the gold tables, load the production model from the **MLflow Model Registry**, and write the 24-hour forecast predictions to a new Delta table.

* **Model Monitoring -- Ongoing**: To ensure the model remains accurate, we'll monitor its performance in production. A scheduled job will compare the model's predictions with the actual carbon intensity values once they become available. This job will calculate key metrics like RMSE and track them over time, allowing us to detect any performance degradation or data drift.

### Technologies Used

* **Databricks**: The unified platform for all data and ML activities.

* **Delta Lake**: The storage layer providing ACID transactions and data versioning.

* **PySpark**: For distributed data processing and transformations.

* **Databricks Jobs**: For scheduling and orchestrating the pipeline stages.

* **Databricks Secrets**: For secure management of API credentials.

* **MLflow**: For end-to-end management of the model lifecycle, from experiment tracking to model registry.

* **Gemini 2.5 Flash**: For efficient code creation of mundane tasks :)