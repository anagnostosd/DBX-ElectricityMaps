from pyspark.sql.functions import col, hour, dayofweek, month, lag, avg, window
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import pyspark.sql.functions as F

# 1. Read data from the silver layer
silver_df = spark.read.format("delta").table("silver.energy_data")

# 2. Handle potential missing values by filling with a reasonable default
# In this case, we'll fill nulls with 0. In a real-world scenario, you might use
# more advanced imputation techniques.
silver_df = silver_df.fillna(0)

# 3. Create time-based features
gold_df = silver_df.withColumn("hour", hour("datetime")) \
                 .withColumn("day_of_week", dayofweek("datetime")) \
                 .withColumn("month", month("datetime"))

# 4. Create time-series features (lagged values and rolling averages)
# We need to use a window function to perform these calculations correctly
window_spec = Window.partitionBy("zone").orderBy("datetime")

# Define the columns for which we want to create lagged features and rolling averages
lag_columns = [
    "consumption_nuclear", "consumption_geothermal", "consumption_biomass",
    "consumption_coal", "consumption_wind", "consumption_solar", "consumption_hydro",
    "consumption_gas", "consumption_oil", "consumption_total",
    "production_nuclear", "production_geothermal", "production_biomass",
    "production_coal", "production_wind", "production_solar", "production_hydro",
    "production_gas", "production_oil", "production_total",
    "import_total", "export_total", "carbon_intensity",
    "import_AL", "import_BG", "import_MK", "import_TR",
    "export_AL", "export_BG", "export_MK", "export_TR"
]

# Add lagged 1-hour features and a 24-hour rolling average for all relevant columns
for column in lag_columns:
    gold_df = gold_df.withColumn(f"lag_{column}_1h", lag(col(column), 1).over(window_spec))
    gold_df = gold_df.withColumn(f"rolling_avg_{column}_24h", avg(col(column)).over(window_spec.rowsBetween(-23, 0)))

# 5. Handle nulls created by the window functions (at the start of the time series)
gold_df = gold_df.fillna(0)

# 6. Write the final DataFrame to the gold layer using an efficient merge strategy
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
gold_table_name = "gold.ml_features"

if not spark.catalog.tableExists(gold_table_name):
    # If the table doesn't exist, create it with a simple write
    print("Gold table does not exist. Creating it...")
    gold_df.write.format("delta").mode("overwrite").saveAsTable(gold_table_name)
else:
    # If the table exists, perform a MERGE
    print("Gold table exists. Merging new data...")
    delta_table = DeltaTable.forName(spark, gold_table_name)
    
    delta_table.alias("old_data") \
        .merge(
            gold_df.alias("new_data"),
            "old_data.zone = new_data.zone AND old_data.datetime = new_data.datetime"
        ) \
        .whenMatchedUpdateAll() \
        .whenNotMatchedInsertAll() \
        .execute()
