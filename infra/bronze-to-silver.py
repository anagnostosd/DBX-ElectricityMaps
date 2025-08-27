from pyspark.sql.functions import from_json, col, current_timestamp, row_number, to_timestamp, lit
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StringType, ArrayType, MapType, LongType, IntegerType, DoubleType, BooleanType
from delta.tables import DeltaTable
from pyspark.sql.functions import from_json, col, current_timestamp, row_number, to_timestamp, lit
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StringType, ArrayType, MapType, LongType, IntegerType, DoubleType, BooleanType

# 1. Read the raw data from the bronze layer, focusing on the most recent data
# We'll read the last 48 hours of ingested data to ensure we capture all recent changes.
power_df_raw = spark.read.format("delta").table("bronze.power_data") \
    .filter("ingested_at >= now() - interval 48 hours")

carbon_df_raw = spark.read.format("delta").table("bronze.carbon_data") \
    .filter("ingested_at >= now() - interval 48 hours")

# 2. Define schemas for the nested JSON to properly parse the data
# Schema for power data
power_schema = StructType([
    StructType([
        StringType(), # zone
        ArrayType(
            StructType([
                StringType(), # zone
                StringType(), # datetime
                StringType(), # updatedAt
                StringType(), # createdAt
                MapType(StringType(), LongType()), # powerConsumptionBreakdown
                MapType(StringType(), LongType()), # powerProductionBreakdown
                MapType(StringType(), LongType()), # powerImportBreakdown
                MapType(StringType(), LongType()), # powerExportBreakdown
                LongType(), # fossilFreePercentage
                LongType(), # renewablePercentage
                LongType(), # powerConsumptionTotal
                LongType(), # powerProductionTotal
                LongType(), # powerImportTotal
                LongType(), # powerExportTotal
                BooleanType(), # isEstimated
                StringType() # estimationMethod
            ])
        ) # history
    ])
])

# Schema for carbon intensity data
carbon_schema = StructType([
    StructType([
        StringType(), # zone
        ArrayType(
            StructType([
                StringType(), # zone
                LongType(), # carbonIntensity
                StringType(), # datetime
                StringType(), # updatedAt
                StringType(), # createdAt
                StringType(), # emissionFactorType
                BooleanType(), # isEstimated
                StringType() # estimationMethod
            ])
        ) # history
    ])
])

# 3. Flatten and select the history array for each dataframe
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

# 4. Deduplicate the flattened data to ensure we have the most recent version
window_spec = Window.partitionBy("zone", "datetime").orderBy(col("ingested_at").desc())

power_df_deduped = power_df_flattened.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

carbon_df_deduped = carbon_df_flattened.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# 5. Join the two dataframes on datetime and zone
silver_df = power_df_deduped.join(carbon_df_deduped, on=["zone", "datetime"], how="inner") \
    .drop(carbon_df_deduped.ingested_at) \
    .drop(power_df_deduped.updatedAt)

# 6. Perform a MERGE to incrementally update the silver table
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
