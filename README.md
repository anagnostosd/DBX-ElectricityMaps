# Electricity Maps Data Pipeline

This project demonstrates a foundational MLOps pipeline on Databricks, transforming raw data from the Electricity Maps and Visual Crossing APIs into a feature-rich, machine-learning-ready gold layer. The pipeline is designed to be incremental, robust, and reproducible, following the core principles of the Medallion Architecture.

## Project Overview

The primary goal of this pipeline is to ingest hourly power and carbon intensity data, as well as weather forecast data for Greece (`GR`). This data is then cleaned, enriched, and prepared for time-series forecasting. The project is structured in three key stages:

* **Bronze Layer**: Raw data ingestion.
* **Silver Layer**: Data cleaning and transformation.
* **Gold Layer**: Feature engineering for machine learning.

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

### Technologies Used

* **Databricks**: The unified platform for all data and ML activities.
* **Delta Lake**: The storage layer providing ACID transactions and data versioning.
* **PySpark**: For distributed data processing and transformations.
* **Databricks Jobs**: For scheduling and orchestrating the pipeline stages.
* **Databricks Secrets**: For secure management of API credentials.