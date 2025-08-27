# Electricity Maps Data Pipeline

This project demonstrates a foundational MLOps pipeline on Databricks, transforming raw data from the Electricity Maps API into a feature-rich, machine-learning-ready gold layer. The pipeline is designed to be incremental, robust, and reproducible, following the core principles of the Medallion Architecture.

## Project Overview

The primary goal of this pipeline is to ingest hourly power and carbon intensity data for Greece (`GR`), clean and enrich it, and prepare it for time-series forecasting. The project is structured in three key stages:

* **Bronze Layer**: Raw data ingestion.

* **Silver Layer**: Data cleaning and transformation.

* **Gold Layer**: Feature engineering for machine learning.

## Pipeline Stages

### 1. Bronze Layer: Raw Data Ingestion

The bronze layer is the entry point for all data. Raw JSON responses from the Electricity Maps API are fetched and stored as a single row in a Delta table.

* **Data Source**: Electricity Maps API v3.

* **Method**: A Python script periodically fetches the latest data.

* **Storage**: Data is stored in two separate Delta tables, `bronze.power_data` and `bronze.carbon_data`. Each table holds the raw JSON string, a source identifier, and a timestamp of ingestion.

* **Key Feature**: This layer is designed for idempotency. The ingestion script appends new data, and the downstream silver layer handles deduplication.

### 2. Silver Layer: Data Cleaning and Transformation

The silver layer takes the raw bronze data, flattens it, and creates a clean, structured table. This is the first step toward creating a reliable "single source of truth."

* **Input**: Raw JSON data from the `bronze.power_data` and `bronze.carbon_data` tables.

* **Transformations**:

  * The nested JSON is parsed and flattened into a structured table.

  * Power and carbon intensity data are joined on `zone` and `datetime`.

  * Deduplication is performed using a window function, keeping only the most recently ingested record for each `datetime`.

* **Output**: A clean, deduplicated Delta table named `silver.energy_data`.

* **Key Feature**: The process uses a Delta Lake `MERGE` statement for efficient, incremental updates, only processing new or updated data. This ensures the silver table is always up-to-date without a full rewrite.

### 3. Gold Layer: Feature Engineering

The gold layer is the final stage, where the silver data is enriched with features ready for direct use in an ML model.

* **Input**: Cleaned and joined data from the `silver.energy_data` table.

* **Features Created**:

  * **Time-based features**: `hour`, `day_of_week`, `month`.

  * **Lagged features**: `lag_..._1h` (1-hour lag) for all key columns (consumption, production, import, export, and carbon intensity).

  * **Rolling averages**: `rolling_avg_..._24h` (24-hour rolling average) for the same key columns.

* **Output**: A single, comprehensive Delta table named `gold.ml_features`.

* **Key Feature**: This layer also uses an incremental `MERGE` strategy, making it an efficient source for continuously updated training data. The data is pre-processed, so a data scientist can start training a model immediately.

### Technologies Used

* **Databricks**: The unified platform for all data and ML activities.

* **Delta Lake**: The storage layer providing ACID transactions and data versioning.

* **PySpark**: For distributed data processing and transformations.

* **Databricks Jobs**: For scheduling and orchestrating the pipeline stages.

* **Databricks Secrets**: For secure management of API credentials.