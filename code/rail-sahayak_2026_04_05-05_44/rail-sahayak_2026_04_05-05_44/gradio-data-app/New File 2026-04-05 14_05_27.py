# ============================================================
# Indian Railways PNR - Databricks ML Pipeline
# Goal: Predict Final Booking Confirmation Status
# ============================================================

# ─────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ─────────────────────────────────────────────
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Imputer
)
from pyspark.ml.classification import (
    RandomForestClassifier, GBTClassifier, LogisticRegression
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

# ─────────────────────────────────────────────
# 1. SCHEMA DEFINITION
# ─────────────────────────────────────────────
schema = StructType([
    StructField("PNR",             StringType(),  True),
    StructField("TrainNo",         IntegerType(), True),
    StructField("BookingDate",     StringType(),  True),   # e.g. "01-Sep"
    StructField("Class",           StringType(),  True),
    StructField("Quota",           StringType(),  True),
    StructField("FromStation",     StringType(),  True),
    StructField("ToStation",       StringType(),  True),
    StructField("TravelDate",      StringType(),  True),   # e.g. "01-Jan"
    StructField("BookingStatus",   StringType(),  True),
    StructField("Passengers",      IntegerType(), True),
    StructField("PassengerType",   StringType(),  True),
    StructField("BookingChannel",  StringType(),  True),
    StructField("Distance",        IntegerType(), True),
    StructField("No of Stations",  IntegerType(), True),
    StructField("Travel Time",     IntegerType(), True),
    StructField("TrainType",       StringType(),  True),
    StructField("Fare",            IntegerType(), True),
    StructField("Concession",      StringType(),  True),
    StructField("Holiday Season",  StringType(),  True),
    StructField("Waitlist No",     StringType(),  True),   # "null" or "WL###"
    StructField("FinalStatus",     StringType(),  True),
])

# ─────────────────────────────────────────────
# 2. INGEST → DELTA LAKE (Bronze Layer)
# ─────────────────────────────────────────────
# FIXED: Use Unity Catalog Volumes instead of DBFS FileStore
RAW_PATH    = "/Volumes/workspace/default/railways/pnr_data.csv"
BRONZE_PATH = "/Volumes/workspace/default/railways/bronze"
SILVER_PATH = "/Volumes/workspace/default/railways/silver"
GOLD_PATH   = "/Volumes/workspace/default/railways/gold"
MODEL_PATH  = "/Volumes/workspace/default/railways/model"
RFR_PATH    = "/Volumes/workspace/default/railways/rfr_results"

# Write inline sample to DBFS (replace with your actual CSV path)
sample_data = """
PNR,TrainNo,BookingDate,Class,Quota,FromStation,ToStation,TravelDate,BookingStatus,Passengers,PassengerType,BookingChannel,Distance,No of Stations,Travel Time,TrainType,Fare,Concession,Holiday Season,Waitlist No,FinalStatus
1234567890,12345,01-Sep,SL,GN,NDLS,BCT,01-Jan,Confirmed,2,Adult,Online,1380,10,18,Express,1200,None,No,null,Confirmed
1234567891,12345,01-Sep,SL,GN,NDLS,BCT,01-Jan,WL,2,Adult,Online,1380,10,18,Express,1200,None,No,WL12,Confirmed
1234567892,12345,01-Sep,SL,GN,NDLS,BCT,01-Jan,WL,2,Adult,Online,1380,10,18,Express,1200,None,No,WL15,Not Confirmed
"""  # Example CSV content

dbutils.fs.put(RAW_PATH, sample_data, overwrite=True)

# Read raw CSV
df_bronze = spark.read.csv(RAW_PATH, header=True, schema=schema)

# Save as Delta Bronze
df_bronze.write.format("delta").mode("overwrite").save(BRONZE_PATH)
spark.sql(f"CREATE TABLE IF NOT EXISTS railways_bronze USING DELTA LOCATION '{BRONZE_PATH}'")
print(f"✅ Bronze layer saved → {BRONZE_PATH}  |  Rows: {df_bronze.count()}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING → SILVER LAYER
# ─────────────────────────────────────────────
df = spark.read.format("delta").load(BRONZE_PATH)

df_silver = df \
    .withColumn("IsWaitlisted",
        F.when(F.col("Waitlist No") == "null", 0).otherwise(1)) \
    .withColumn("WaitlistRank",
        F.when(F.col("Waitlist No") == "null", 0)
         .otherwise(F.regexp_extract("Waitlist No", r"WL(\d+)", 1).cast("int"))) \
    .withColumn("IsHoliday",
        F.when(F.col("Holiday Season") == "Yes", 1).otherwise(0)) \
    .withColumn("HasConcession",
        F.when(F.col("Concession") == "None", 0).otherwise(1)) \
    .withColumn("IsConfirmed_AtBooking",
        F.when(F.col("BookingStatus") == "Confirmed", 1).otherwise(0)) \
    .withColumn("label",                                   # ← prediction target
        F.when(F.col("FinalStatus") == "Confirmed", 1.0).otherwise(0.0)) \
    .dropna(subset=["label", "Distance", "Fare", "Passengers"])

df_silver.write.format("delta").mode("overwrite").save(SILVER_PATH)
spark.sql(f"CREATE TABLE IF NOT EXISTS railways_silver USING DELTA LOCATION '{SILVER_PATH}'")
print(f"✅ Silver layer saved → {SILVER_PATH}")

# ─────────────────────────────────────────────
# 4. GOLD LAYER — AGGREGATED FEATURES
# ─────────────────────────────────────────────
df_gold = spark.read.format("delta").load(SILVER_PATH) \
    .groupBy("FromStation", "ToStation", "Class", "TrainType") \
    .agg(
        F.avg("label").alias("route_confirmation_rate"),
        F.avg("Fare").alias("avg_fare_on_route"),
        F.avg("Distance").alias("avg_distance"),
        F.count("PNR").alias("bookings_count")
    )

df_gold.write.format("delta").mode("overwrite").save(GOLD_PATH)
spark.sql(f"CREATE TABLE IF NOT EXISTS railways_gold USING DELTA LOCATION '{GOLD_PATH}'")
print(f"✅ Gold layer saved → {GOLD_PATH}")

# Join Gold route stats back into Silver for enriched training set
df_train = spark.read.format("delta").load(SILVER_PATH) \
    .join(df_gold, on=["FromStation", "ToStation", "Class", "TrainType"], how="left")

# ─────────────────────────────────────────────
# 5. ML PIPELINE DEFINITION
# ─────────────────────────────────────────────
CATEGORICAL_COLS = [
    "Class", "Quota", "FromStation", "ToStation",
    "BookingStatus", "PassengerType", "BookingChannel",
    "TrainType", "Concession"
]
NUMERIC_COLS = [
    "Distance", "Fare", "Passengers",
    "IsWaitlisted", "WaitlistRank",
    "IsHoliday", "HasConcession", "IsConfirmed_AtBooking",
    "route_confirmation_rate", "avg_fare_on_route",
    "avg_distance", "bookings_count"
]

# Categorical → Index → OHE
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in CATEGORICAL_COLS
]
encoders = [
    OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
    for c in CATEGORICAL_COLS
]

# Numeric imputer (fill nulls with median)
imputer = Imputer(
    inputCols=NUMERIC_COLS,
    outputCols=[f"{c}_imp" for c in NUMERIC_COLS],
    strategy="median"
)

# Assemble all features
feature_cols = (
    [f"{c}_ohe" for c in CATEGORICAL_COLS] +
    [f"{c}_imp" for c in NUMERIC_COLS]
)
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="keep"
)
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True, withStd=True
)

# Three candidate classifiers
rf  = RandomForestClassifier(featuresCol="features", labelCol="label",
                              numTrees=100, maxDepth=8, seed=42)
gbt = GBTClassifier(featuresCol="features", labelCol="label",
                    maxIter=50, maxDepth=6, seed=42)
lr  = LogisticRegression(featuresCol="features", labelCol="label",
                         maxIter=100, regParam=0.01)

# ─────────────────────────────────────────────
# 6. TRAIN / EVAL WITH MLFLOW
# ─────────────────────────────────────────────
train_df, test_df = df_train.randomSplit([0.8, 0.2], seed=42)

mlflow.set_experiment("/Shared/RailwaysPNR_Confirmation")

def train_and_log(model, model_name: str):
    """Train a full pipeline, evaluate, log everything to MLflow."""
    pipeline = Pipeline(stages=indexers + encoders + [imputer, assembler, scaler, model])

    with mlflow.start_run(run_name=model_name):
        # ── Hyperparameter grid ──────────────────────
        if isinstance(model, RandomForestClassifier):
            param_grid = ParamGridBuilder() \
                .addGrid(model.numTrees, [50, 100]) \
                .addGrid(model.maxDepth, [6, 8]) \
                .build()
        elif isinstance(model, GBTClassifier):
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxIter, [30, 50]) \
                .addGrid(model.stepSize, [0.05, 0.1]) \
                .build()
        else:
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam, [0.001, 0.01]) \
                .build()

        auc_eval = BinaryClassificationEvaluator(
            labelCol="label", metricName="areaUnderROC"
        )
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=auc_eval,
            numFolds=3,
            seed=42
        )

        cv_model   = cv.fit(train_df)
        best_model = cv_model.bestModel
        preds      = best_model.transform(test_df)

        # ── Metrics ─────────────────────────────────
        auc      = auc_eval.evaluate(preds)
        acc_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        f1_eval  = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )
        accuracy = acc_eval.evaluate(preds)
        f1       = f1_eval.evaluate(preds)

        # ── Log to MLflow ────────────────────────────
        mlflow.log_metric("auc_roc",  auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log best params
        best_stage = best_model.stages[-1]
        for k, v in best_stage.extractParamMap().items():
            mlflow.log_param(k.name, v)

        # Log model artifact
        signature = infer_signature(
            train_df.drop("label").limit(5).toPandas(),
            preds.select("prediction").limit(5).toPandas()
        )
        mlflow.spark.log_model(
            best_model,
            artifact_path=model_name,
            signature=signature,
            registered_model_name=f"RailwaysPNR_{model_name}"
        )

        print(f"  [{model_name}]  AUC={auc:.4f}  Acc={accuracy:.4f}  F1={f1:.4f}")
        return best_model, auc

print("\n🚂  Training models …")
rf_model,  rf_auc  = train_and_log(rf,  "RandomForest")
gbt_model, gbt_auc = train_and_log(gbt, "GBT")
lr_model,  lr_auc  = train_and_log(lr,  "LogisticRegression")

# Select best model
best = max([(rf_model, rf_auc, "RF"),
            (gbt_model, gbt_auc, "GBT"),
            (lr_model,  lr_auc,  "LR")],
           key=lambda x: x[1])
best_model, best_auc, best_name = best
print(f"\n🏆  Best model: {best_name}  (AUC = {best_auc:.4f})")

# ─────────────────────────────────────────────
# 7. RANDOM FOREST REGRESSOR — Predict Fare
# ─────────────────────────────────────────────
print("\n🌲  Training RandomForestRegressor to predict Fare …")

# Regression feature set — exclude Fare from inputs, use it as target
RFR_NUMERIC_COLS = [
    "Distance", "Passengers",
    "IsWaitlisted", "WaitlistRank",
    "IsHoliday", "HasConcession", "IsConfirmed_AtBooking",
    "route_confirmation_rate", "avg_fare_on_route",
    "avg_distance", "bookings_count"
]
RFR_CATEGORICAL_COLS = [
    "Class", "Quota", "FromStation", "ToStation",
    "BookingStatus", "PassengerType", "BookingChannel",
    "TrainType", "Concession"
]

# Re-use same indexer/encoder/imputer pattern for regression features
rfr_indexers = [
    StringIndexer(inputCol=c, outputCol=f"rfr_{c}_idx", handleInvalid="keep")
    for c in RFR_CATEGORICAL_COLS
]
rfr_encoders = [
    OneHotEncoder(inputCol=f"rfr_{c}_idx", outputCol=f"rfr_{c}_ohe")
    for c in RFR_CATEGORICAL_COLS
]
rfr_imputer = Imputer(
    inputCols=RFR_NUMERIC_COLS,
    outputCols=[f"rfr_{c}_imp" for c in RFR_NUMERIC_COLS],
    strategy="median"
)
rfr_feature_cols = (
    [f"rfr_{c}_ohe" for c in RFR_CATEGORICAL_COLS] +
    [f"rfr_{c}_imp" for c in RFR_NUMERIC_COLS]
)
rfr_assembler = VectorAssembler(
    inputCols=rfr_feature_cols,
    outputCol="rfr_raw_features",
    handleInvalid="keep"
)
rfr_scaler = StandardScaler(
    inputCol="rfr_raw_features",
    outputCol="rfr_features",
    withMean=True, withStd=True
)

# RandomForestRegressor — predict Fare
rfr = RandomForestRegressor(
    featuresCol="rfr_features",
    labelCol="Fare",
    numTrees=100,
    maxDepth=8,
    seed=42
)

rfr_pipeline = Pipeline(
    stages=rfr_indexers + rfr_encoders +
           [rfr_imputer, rfr_assembler, rfr_scaler, rfr]
)

# Hyperparameter tuning via CrossValidator
rfr_param_grid = ParamGridBuilder() \
    .addGrid(rfr.numTrees,  [50, 100]) \
    .addGrid(rfr.maxDepth,  [6, 8]) \
    .addGrid(rfr.minInstancesPerNode, [1, 2]) \
    .build()

rfr_evaluator = RegressionEvaluator(
    labelCol="Fare",
    predictionCol="prediction",
    metricName="rmse"
)

rfr_cv = CrossValidator(
    estimator=rfr_pipeline,
    estimatorParamMaps=rfr_param_grid,
    evaluator=rfr_evaluator,
    numFolds=3,
    seed=42
)

# Train/test split on Silver + Gold enriched data
rfr_train, rfr_test = df_train.randomSplit([0.8, 0.2], seed=42)

with mlflow.start_run(run_name="RandomForestRegressor_Fare"):
    rfr_cv_model   = rfr_cv.fit(rfr_train)
    rfr_best_model = rfr_cv_model.bestModel
    rfr_preds      = rfr_best_model.transform(rfr_test)

    # ── Regression Metrics ───────────────────────
    rmse = rfr_evaluator.evaluate(rfr_preds)

    mae_eval = RegressionEvaluator(
        labelCol="Fare", predictionCol="prediction", metricName="mae"
    )
    r2_eval = RegressionEvaluator(
        labelCol="Fare", predictionCol="prediction", metricName="r2"
    )
    mae = mae_eval.evaluate(rfr_preds)
    r2  = r2_eval.evaluate(rfr_preds)

    # ── Feature Importance ───────────────────────
    rfr_stage       = rfr_best_model.stages[-1]
    feat_importances = rfr_stage.featureImportances

    # ── Log to MLflow ────────────────────────────
    mlflow.log_metric("rmse",   rmse)
    mlflow.log_metric("mae",    mae)
    mlflow.log_metric("r2",     r2)

    best_params = rfr_stage.extractParamMap()
    for k, v in best_params.items():
        mlflow.log_param(k.name, v)

    rfr_signature = infer_signature(
        rfr_train.drop("Fare").limit(5).toPandas(),
        rfr_preds.select("prediction").limit(5).toPandas()
    )
    mlflow.spark.log_model(
        rfr_best_model,
        artifact_path="RandomForestRegressor",
        signature=rfr_signature,
        registered_model_name="RailwaysPNR_FareRegressor"
    )

    print(f"  [RandomForestRegressor]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

# Save regression predictions to Delta
rfr_preds.select("PNR", "Fare", "prediction") \
    .withColumnRenamed("prediction", "predicted_fare") \
    .write.format("delta").mode("overwrite").save(RFR_PATH)

print(f"✅ RFR predictions saved → {RFR_PATH}")
display(rfr_preds.select("PNR", "Fare", "predicted_fare"))

# ─────────────────────────────────────────────
# 8. BATCH INFERENCE + SAVE RESULTS
# ─────────────────────────────────────────────
print("\n🤖  Running batch inference (classification) …")
inference_df = best_model.transform(test_df)
results_df   = inference_df.select(
    "PNR", "FinalStatus", "label",
    "prediction", "probability"
)

results_df.write.format("delta") \
    .mode("overwrite") \
    .save("/Volumes/workspace/default/railways/inference_results")

display(results_df)

# ─────────────────────────────────────────────
# 9. MODEL SERVING ENDPOINT (Databricks)
# ─────────────────────────────────────────────
# Uncomment once you're ready to deploy via REST API

# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()
# w.serving_endpoints.create(
#     name="railways-pnr-confirmation",
#     config={
#         "served_models": [{
#             "model_name":    f"RailwaysPNR_{best_name}",
#             "model_version": "1",
#             "workload_size": "Small",
#             "scale_to_zero_enabled": True,
#         }]
#     }
# )

# ─────────────────────────────────────────────
# 10. DELTA LAKE VERSION HISTORY
# ─────────────────────────────────────────────
print("\n📜  Delta Lake version history (Silver):")
display(
    spark.sql(f"DESCRIBE HISTORY delta.`{SILVER_PATH}`")
     .select("version", "timestamp", "operation")
     .limit(5)
)

print("\n✅  Pipeline complete.")
