# ============================================================
# Indian Railways PNR - Databricks ML Pipeline
# Goal: Predict Final Booking Confirmation Status
# ============================================================

# ─────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ─────────────────────────────────────────────
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Imputer
)
from pyspark.ml.classification import (
    RandomForestClassifier, GBTClassifier, LogisticRegression
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

# ─────────────────────────────────────────────
# 1. INGEST FROM UNITY CATALOG TABLE
# ─────────────────────────────────────────────
SOURCE_TABLE = "workspace.default.railway_ticket_confirmation"
SILVER_TABLE = "workspace.default.railways_silver"
GOLD_TABLE = "workspace.default.railways_gold"

print(f"📊 Reading data from {SOURCE_TABLE}...")
df_bronze = spark.table(SOURCE_TABLE)
print(f"✅ Loaded {df_bronze.count()} rows")

# Show schema and sample
df_bronze.printSchema()
display(df_bronze.limit(5))

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING → SILVER LAYER
# ─────────────────────────────────────────────
print("\n🔧 Engineering features...")

df_silver = df_bronze \
    .withColumn("IsWaitlisted",
        F.when(F.col("`Waitlist Position`").isNull(), 0)
         .when(F.col("`Waitlist Position`") == "null", 0)
         .otherwise(1)) \
    .withColumn("WaitlistRank",
        F.when(F.col("`Waitlist Position`").isNull(), 0)
         .when(F.col("`Waitlist Position`") == "null", 0)
         .otherwise(F.regexp_extract("`Waitlist Position`", r"WL(\d+)", 1).cast("int"))) \
    .withColumn("IsHoliday",
        F.when(F.col("`Holiday or Peak Season`") == "Yes", 1).otherwise(0)) \
    .withColumn("HasConcession",
        F.when(F.col("`Special Considerations`") == "None", 0)
         .when(F.col("`Special Considerations`").isNull(), 0)
         .otherwise(1)) \
    .withColumn("IsConfirmed_AtBooking",
        F.when(F.col("`Current Status`") == "Confirmed", 1).otherwise(0)) \
    .withColumn("label",
        F.when(F.col("`Confirmation Status`") == "Confirmed", 1.0).otherwise(0.0)) \
    .dropna(subset=["label", "`Travel Distance`", "`Number of Passengers`"])

# Rename all columns to replace spaces with underscores (Delta Lake requirement)
for col_name in df_silver.columns:
    new_col_name = col_name.replace(" ", "_")
    df_silver = df_silver.withColumnRenamed(col_name, new_col_name)

print(f"✅ Columns renamed to remove spaces")

# Save as managed table
df_silver.write.format("delta").mode("overwrite").saveAsTable(SILVER_TABLE)
print(f"✅ Silver layer saved → {SILVER_TABLE}  |  Rows: {df_silver.count()}")

# ─────────────────────────────────────────────
# 3. GOLD LAYER — AGGREGATED FEATURES
# ─────────────────────────────────────────────
print("\n📈 Creating aggregated features...")

df_gold = spark.table(SILVER_TABLE) \
    .groupBy("Source_Station", "Destination_Station", "Class_of_Travel", "Train_Type") \
    .agg(
        F.avg("label").alias("route_confirmation_rate"),
        F.avg("Travel_Distance").alias("avg_distance"),
        F.count("PNR_Number").alias("bookings_count")
    )

df_gold.write.format("delta").mode("overwrite").saveAsTable(GOLD_TABLE)
print(f"✅ Gold layer saved → {GOLD_TABLE}")

# Join Gold route stats back into Silver for enriched training set
df_train = spark.table(SILVER_TABLE) \
    .join(df_gold, 
          on=["Source_Station", "Destination_Station", "Class_of_Travel", "Train_Type"], 
          how="left")

print(f"\n✅ Training dataset ready: {df_train.count()} rows")
display(df_train.limit(5))