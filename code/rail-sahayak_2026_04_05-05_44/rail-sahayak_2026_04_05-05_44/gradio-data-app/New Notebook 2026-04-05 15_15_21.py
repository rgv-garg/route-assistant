# Databricks notebook source
# Export station data for train 12673 (and all other trains)
from pyspark.sql import SparkSession

# Query your table that has train and station data
station_df = spark.table('workspace.default.train_delay_facts') \
    .select('train_number', 'station') \
    .distinct() \
    .toPandas()

# Save to app directory
output_path = '/Workspace/Users/tejask@iisc.ac.in/databricks_apps/rail-sahayak_2026_04_05-05_44/gradio-data-app/station_lookup.csv'
station_df.to_csv(output_path, index=False)

# Verify
print(f"✓ Exported {len(station_df)} station records")
print(f"✓ Unique trains: {station_df['train_number'].nunique()}")

# Check if train 12673 is included
train_12673 = station_df[station_df['train_number'] == 12673]
print(f"\n✓ Train 12673: {len(train_12673)} stations")
print(train_12673.head())