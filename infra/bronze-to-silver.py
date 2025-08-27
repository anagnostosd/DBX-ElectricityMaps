from pyspark.sql.functions import from_json, col, current_timestamp, row_number, to_timestamp, lit
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StringType, ArrayType, MapType, LongType, IntegerType, DoubleType, BooleanType, StructField
from delta.tables import DeltaTable

# 1. Get the lookback interval from a notebook widget. Default to 48 hours.
try:
    lookback_hours = int(dbutils.widgets.get("lookback_hours"))
except:
    lookback_hours = 48
    print(f"No lookback_hours parameter provided, defaulting to {lookback_hours} hours.")

# 2. Read the raw data from the bronze layer, focusing on the most recent data
# We'll read data ingested within the specified lookback period to capture all recent changes.
power_df_raw = spark.read.format("delta").table("bronze.power_data") \
    .filter(f"ingested_at >= now() - interval {lookback_hours} hours")

carbon_df_raw = spark.read.format("delta").table("bronze.carbon_data") \
    .filter(f"ingested_at >= now() - interval {lookback_hours} hours")

# 3. Define schemas for the nested JSON to properly parse the data
# Correct Schema for power data, including the root-level fields in the correct order
power_schema = StructType([
    StructField("zone", StringType()),
    StructField("history", ArrayType(
        StructType([
            StructField("zone", StringType()),
            StructField("datetime", StringType()),
            StructField("updatedAt", StringType()),
            StructField("createdAt", StringType()),
            StructField("powerConsumptionBreakdown", MapType(StringType(), LongType())),
            StructField("powerProductionBreakdown", MapType(StringType(), LongType())),
            StructField("powerImportBreakdown", MapType(StringType(), LongType())),
            StructField("powerExportBreakdown", MapType(StringType(), LongType())),
            StructField("fossilFreePercentage", LongType()),
            StructField("renewablePercentage", LongType()),
            StructField("powerConsumptionTotal", LongType()),
            StructField("powerProductionTotal", LongType()),
            StructField("powerImportTotal", LongType()),
            StructField("powerExportTotal", LongType()),
            StructField("isEstimated", BooleanType()),
            StructField("estimationMethod", StringType())
        ])
    )),
    StructField("temporalGranularity", StringType())
])

# Correct Schema for carbon intensity data, including the root-level fields in the correct order
carbon_schema = StructType([
    StructField("zone", StringType()),
    StructField("history", ArrayType(
        StructType([
            StructField("zone", StringType()),
            StructField("carbonIntensity", LongType()),
            StructField("datetime", StringType()),
            StructField("updatedAt", StringType()),
            StructField("createdAt", StringType()),
            StructField("emissionFactorType", StringType()),
            StructField("isEstimated", BooleanType()),
            StructField("estimationMethod", StringType())
        ])
    )),
    StructField("temporalGranularity", StringType())
])

# 4. Flatten and select the history array for each dataframe
# We will explode the 'history' array to create a row for each entry
power_df_flattened = (
    power_df_raw.select(
        from_json(col("raw_json"), power_schema).alias("json_data"), 
        col("ingested_at")
    )
    .select(col("json_data.history"), col("ingested_at"))
    .withColumn("history", col("history"))
    .selectExpr("ingested_at", "explode(history) as history_item")
    .selectExpr(
        "history_item.zone",
        "to_timestamp(history_item.datetime, 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z') as datetime",
        "to_timestamp(history_item.updatedAt, 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z') as updatedAt",
        "history_item.isEstimated",
        "history_item.powerConsumptionBreakdown.nuclear as consumption_nuclear",
        "history_item.powerConsumptionBreakdown.geothermal as consumption_geothermal",
        "history_item.powerConsumptionBreakdown.biomass as consumption_biomass",
        "history_item.powerConsumptionBreakdown.coal as consumption_coal",
        "history_item.powerConsumptionBreakdown.wind as consumption_wind",
        "history_item.powerConsumptionBreakdown.solar as consumption_solar",
        "history_item.powerConsumptionBreakdown.hydro as consumption_hydro",
        "history_item.powerConsumptionBreakdown.gas as consumption_gas",
        "history_item.powerConsumptionBreakdown.oil as consumption_oil",
        "history_item.powerConsumptionTotal as consumption_total",
        "history_item.powerProductionBreakdown.nuclear as production_nuclear",
        "history_item.powerProductionBreakdown.geothermal as production_geothermal",
        "history_item.powerProductionBreakdown.biomass as production_biomass",
        "history_item.powerProductionBreakdown.coal as production_coal",
        "history_item.powerProductionBreakdown.wind as production_wind",
        "history_item.powerProductionBreakdown.solar as production_solar",
        "history_item.powerProductionBreakdown.hydro as production_hydro",
        "history_item.powerProductionBreakdown.gas as production_gas",
        "history_item.powerProductionBreakdown.oil as production_oil",
        "history_item.powerProductionTotal as production_total",
        "history_item.powerImportTotal as import_total",
        "history_item.powerExportTotal as export_total",
        "history_item.powerImportBreakdown.AL as import_AL",
        "history_item.powerImportBreakdown.BG as import_BG",
        "history_item.powerImportBreakdown.MK as import_MK",
        "history_item.powerImportBreakdown.TR as import_TR",
        "history_item.powerExportBreakdown.AL as export_AL",
        "history_item.powerExportBreakdown.BG as export_BG",
        "history_item.powerExportBreakdown.MK as export_MK",
        "history_item.powerExportBreakdown.TR as export_TR"
    )
)

carbon_df_flattened = (
    carbon_df_raw.select(
        from_json(col("raw_json"), carbon_schema).alias("json_data"), 
        col("ingested_at")
    )
    .select(col("json_data.history"), col("ingested_at"))
    .withColumn("history", col("history"))
    .selectExpr("ingested_at", "explode(history) as history_item")
    .selectExpr(
        "history_item.zone",
        "to_timestamp(history_item.datetime, 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z') as datetime",
        "to_timestamp(history_item.updatedAt, 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z') as updatedAt",
        "history_item.isEstimated",
        "history_item.carbonIntensity as carbon_intensity"
    )
)

# 5. Deduplicate the flattened data to ensure we have the most recent version
window_spec = Window.partitionBy("zone", "datetime").orderBy(col("ingested_at").desc())

power_df_deduped = power_df_flattened.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

carbon_df_deduped = carbon_df_flattened.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# 6. Join the two dataframes on datetime and zone
silver_df = power_df_deduped.join(carbon_df_deduped, on=["zone", "datetime"], how="inner") \
    .drop(carbon_df_deduped.ingested_at) \
    .drop(power_df_deduped.updatedAt)

# 7. Perform a MERGE to incrementally update the silver table
# This is the key to an efficient, idempotent pipeline
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
silver_table_path = "silver.energy_data"

# Check if the silver table already exists
if not spark.catalog._jcatalog.tableExists(silver_table_path):
    # If the table doesn't exist, create it with a simple write
    print("Silver table does not exist. Creating it...")
    silver_df.write.format("delta").mode("overwrite").saveAsTable(silver_table_path)
else:
    # If the table exists, perform a MERGE
    print("Silver table exists. Merging new data...")
    delta_table = DeltaTable.forName(spark, silver_table_path)
    
    delta_table.alias("old_data") \
        .merge(
            silver_df.alias("new_data"),
            "old_data.zone = new_data.zone AND old_data.datetime = new_data.datetime"
        ) \
        .whenMatchedUpdateAll() \
        .whenNotMatchedInsertAll() \
        .execute()
        
print("Silver layer update complete.")
