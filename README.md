2. Silver Layer: Data Cleaning and Transformation

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

The gold layer is the final stage, where the silver data is enriched with features ready for direct use in an ML model.

* **Input**: Cleaned data from both `silver.energy_data` and `silver.weather_forecast_data` tables.

* **Transformations**:

    * The weather data is first pivoted from a long format (one row per city) to a wide format (columns for each city's weather metrics).

    * This pivoted data is joined with the energy data on `datetime`.

* **Features Created**:

    * **Time-based features**: `hour`, `day_of_week`, `month`.

    * **Lagged features**: `lag_..._1h` (1-hour lag) for all key columns (consumption, production, import, export, carbon intensity, and weather data).

    * **Rolling averages**: `rolling_avg_..._24h` (24-hour rolling average) for the same key columns.

* **Output**: A single, comprehensive Delta table named `gold.ml_features`.

* **Key Feature**: This layer also uses an incremental `MERGE` strategy, making it an efficient source for continuously updated training data. The data is pre-processed, so a data scientist can start training a model immediately.

### Technologies Used

* **Databricks**: The unified platform for all data and ML activities.

* **Delta Lake**: The storage layer providing ACID transactions and data versioning.

* **PySpark**: For distributed data processing and transformations.

* **Databricks Jobs**: For scheduling and orchestrating the pipeline stages.

* **Databricks Secrets**: For secure management of API credentials.