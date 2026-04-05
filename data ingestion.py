import json
import calendar
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# ── Config ───────────────────────────────────────────────────────────────────
JSON_PATH = '/Volumes/workspace/default/raildelaydata/railDelayData.json'
CATALOG   = 'workspace'
DATABASE  = 'default'
TABLE     = 'train_delay_facts'

# ── Read & flatten ───────────────────────────────────────────────────────────
raw_text = dbutils.fs.head(JSON_PATH, 50_000_000)
trains   = json.loads(raw_text)

rows = []
skipped = 0
for train in trains:
    train_no   = int(train['Number'])
    train_name = train['Name']
    data_rows  = train['data']

    stations = [
        col['label'] if isinstance(col, dict) else str(col)
        for col in data_rows[0][1:]
    ]

    for row in data_rows[1:]:
        parts = str(row[0]).split(',')
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if day > calendar.monthrange(year, month)[1]:  
                skipped += 1
                continue
        except (ValueError, IndexError):
            skipped += 1
            continue

        for station, delay in zip(stations, row[1:]):
            rows.append((
                train_no, train_name,
                year, month, day,
                station,
                int(delay) if delay is not None else None
            ))

print(f'Rows parsed  : {len(rows):,}')
print(f'Rows skipped : {skipped}')

# ── Create DataFrame ─────────────────────────────────────────────────────────
schema = StructType([
    StructField('train_number', IntegerType(), False),
    StructField('train_name',   StringType(),  False),
    StructField('year',         IntegerType(), False),
    StructField('month',        IntegerType(), False),
    StructField('day',          IntegerType(), False),
    StructField('station',      StringType(),  False),
    StructField('delay_min',    IntegerType(), True),
])

df = (
    spark.createDataFrame(rows, schema=schema)
    .withColumn('travel_date', F.make_date('year', 'month', 'day'))
    .drop('year', 'month', 'day')
    .withColumn('ingested_at', F.current_timestamp())
)

print(f'Records in DataFrame: {df.count():,}')
display(df)

# ── Write raw to Delta Lake ───────────────────────────────────────────────────
(
    df.write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .partitionBy('travel_date')
    .saveAsTable(f'{CATALOG}.{DATABASE}.{TABLE}')
)

print(f'✅  Raw data in Delta: {CATALOG}.{DATABASE}.{TABLE}')

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = 'workspace'
DATABASE = 'default'
TABLE   = 'train_delay_facts'

df = spark.table(f'{CATALOG}.{DATABASE}.{TABLE}')

# ── 1. Weekend / Weekday / Holiday flag ───────────────────────────────────────
INDIAN_HOLIDAYS = [
    '2025-01-26', '2025-03-17', '2025-04-14', '2025-04-18',
    '2025-08-15', '2025-10-02', '2025-10-20', '2025-11-05', '2025-12-25'
]

df = df.withColumn(
    'day_type',
    F.when(
        F.date_format('travel_date', 'yyyy-MM-dd').isin(INDIAN_HOLIDAYS), 2  # holiday
    ).when(
        F.dayofweek('travel_date').isin([1, 7]), 1                           # weekend
    ).otherwise(0)                                                            # weekday
)
# 0 = Weekday, 1 = Weekend, 2 = Holiday

# ── 2. Rolling 72hr avg delay per (station) — maintenance inference ───────────
# Window: per station, ordered by date, last 3 days = 72 hours
# shift(1) ensures we never include current row (no data leakage)
window_72hr = (
    Window
    .partitionBy('station')
    .orderBy(F.unix_date('travel_date'))          # ← unix_date for DATE type
    .rangeBetween(-3, -1)                          # ← 3 days back (unix_date is in days not seconds)
)

df = df.withColumn(
    'rolling_72hr_avg_delay',
    F.avg('delay_min').over(window_72hr)
)

# Fill nulls (first few days won't have 72hr history)
df = df.fillna({'rolling_72hr_avg_delay': 0})

# ── 3. Maintenance likely flag ────────────────────────────────────────────────
df = df.withColumn(
    'is_maintenance_likely',
    F.when(F.col('rolling_72hr_avg_delay') > 15, 1).otherwise(0)
)

# ── Save back to Delta ────────────────────────────────────────────────────────
(
    df.write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable(f'{CATALOG}.{DATABASE}.{TABLE}')
)

print('✅  Added: day_type, rolling_72hr_avg_delay, is_maintenance_likely')
display(
    df.select(
        'travel_date', 'station', 'delay_min',
        'day_type', 'rolling_72hr_avg_delay', 'is_maintenance_likely'
    ).limit(20)
)

from pyspark.sql import functions as F
from pyspark.sql.types import *

# ── Config ────────────────────────────────────────────────────────────────────
WEATHER_CSV_PATH = '/Volumes/workspace/default/raildelaydata/Weather data (half).csv'
CATALOG          = 'workspace'
DATABASE         = 'default'
WEATHER_TABLE    = 'weather_facts'

# ── Read CSV ──────────────────────────────────────────────────────────────────
weather_df = (
    spark.read
    .option('header', 'true')
    .option('inferSchema', 'true')
    .csv(WEATHER_CSV_PATH)
)

print('Raw schema:')
weather_df.printSchema()
display(weather_df.limit(5))

# ── Clean & standardise columns ───────────────────────────────────────────────
weather_df = (
    weather_df
    .withColumnRenamed('Date',                    'weather_date')
    .withColumnRenamed('Latitude',                'latitude')
    .withColumnRenamed('Longitude',               'longitude')
    .withColumnRenamed('weather_code (wmo code)', 'weather_code')
    .withColumnRenamed('precipitation_sum (mm)',  'precipitation_mm')
    .withColumn('weather_date', F.to_date('weather_date', 'dd-MM-yyyy'))
)

# ── Add weather risk score ────────────────────────────────────────────────────
weather_df = weather_df.withColumn(
    'weather_risk_score',
    F.when(F.col('weather_code') == 0,                0)  # clear
     .when(F.col('weather_code').isin([1, 2, 3]),     1)  # cloudy
     .when(F.col('weather_code').isin([51, 53, 55]),  2)  # drizzle
     .when(F.col('weather_code').isin([61, 63, 65]),  3)  # rain
     .when(F.col('weather_code').isin([80, 81, 82]),  3)  # showers
     .when(F.col('weather_code').isin([45, 48]),      3)  # fog
     .when(F.col('weather_code').isin([95, 96, 99]),  4)  # thunderstorm
     .otherwise(1)
)

# ── Map to nearest station via rounded coordinates ────────────────────────────
station_master = spark.table(f'{CATALOG}.{DATABASE}.station_master')

weather_df = weather_df \
    .withColumn('lat_round', F.round('latitude',  2)) \
    .withColumn('lon_round', F.round('longitude', 2))

station_master = station_master \
    .withColumn('lat_round', F.round('latitude',  2)) \
    .withColumn('lon_round', F.round('longitude', 2))

weather_with_station = (
    weather_df
    .join(
        station_master.select('station_code', 'station_name', 'lat_round', 'lon_round'),
        on=['lat_round', 'lon_round'],
        how='left'
    )
    .drop('lat_round', 'lon_round')
)

print(f'Total weather records    : {weather_with_station.count():,}')
print(f'Stations matched         : {weather_with_station.filter(F.col("station_code").isNotNull()).count():,}')
print(f'Unmatched records        : {weather_with_station.filter(F.col("station_code").isNull()).count():,}')
display(weather_with_station.limit(10))

# ── Write to Delta Lake ───────────────────────────────────────────────────────
(
    weather_with_station.write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .partitionBy('weather_date')
    .saveAsTable(f'{CATALOG}.{DATABASE}.{WEATHER_TABLE}')
)

print(f'✅  Weather facts saved to {CATALOG}.{DATABASE}.{WEATHER_TABLE}')

# ── Verify train route station coverage ───────────────────────────────────────
train_stations = ['CBE', 'MAS', 'AJJ', 'KPD', 'JTJ', 'SA', 'ED', 'TUP', 'CBF']

print('\n🔍 Weather coverage for train route stations:')
spark.table(f'{CATALOG}.{DATABASE}.{WEATHER_TABLE}') \
    .filter(F.col('station_code').isin(train_stations)) \
    .groupBy('station_code') \
    .agg(
        F.count('*').alias('weather_records'),
        F.min('weather_date').alias('from_date'),
        F.max('weather_date').alias('to_date')
    ) \
    .orderBy('station_code') \
    .display()

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = 'workspace'
DATABASE = 'default'

# ── Load both tables ──────────────────────────────────────────────────────────
delays  = spark.table(f'{CATALOG}.{DATABASE}.train_delay_facts')
weather = spark.table(f'{CATALOG}.{DATABASE}.weather_facts') \
    .select('station_code', 'weather_date', 'weather_code',
            'precipitation_mm', 'weather_risk_score')

# ── Join weather onto delay facts ─────────────────────────────────────────────
joined = delays.join(
    weather,
    on=[
        delays.station     == weather.station_code,
        delays.travel_date == weather.weather_date
    ],
    how='left'
).drop('station_code', 'weather_date')

# ── For missing weather — fill with daily avg across matched stations ──────────
# Compute avg weather per date across all matched stations
daily_avg_weather = (
    weather
    .groupBy('weather_date')
    .agg(
        F.avg('precipitation_mm').alias('avg_precipitation_mm'),
        F.avg('weather_risk_score').alias('avg_weather_risk_score'),
        F.avg('weather_code').alias('avg_weather_code'),
    )
)

# Join daily avg onto main table
joined = joined.join(
    daily_avg_weather,
    on=[joined.travel_date == daily_avg_weather.weather_date],
    how='left'
).drop('weather_date')

# ── Fill nulls with daily averages ────────────────────────────────────────────
joined = (
    joined
    .withColumn('precipitation_mm',
        F.when(F.col('precipitation_mm').isNull(),
               F.col('avg_precipitation_mm'))
         .otherwise(F.col('precipitation_mm')))
    .withColumn('weather_risk_score',
        F.when(F.col('weather_risk_score').isNull(),
               F.col('avg_weather_risk_score'))
         .otherwise(F.col('weather_risk_score')))
    .withColumn('weather_code',
        F.when(F.col('weather_code').isNull(),
               F.col('avg_weather_code'))
         .otherwise(F.col('weather_code')))
    # Fill any remaining nulls with 0
    .fillna({
        'precipitation_mm':   0.0,
        'weather_risk_score': 0.0,
        'weather_code':       0.0
    })
    .drop('avg_precipitation_mm', 'avg_weather_risk_score', 'avg_weather_code')
)

# ── Verify coverage after fill ────────────────────────────────────────────────
total        = joined.count()
with_weather = joined.filter(F.col('weather_risk_score').isNotNull()).count()
print(f'Coverage after fill: {round(with_weather/total*100, 1)}%')  # should be 100%

# ── Save back to train_delay_facts with weather columns ───────────────────────
(
    joined.write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .partitionBy('travel_date')
    .saveAsTable(f'{CATALOG}.{DATABASE}.train_delay_facts')
)

print('✅  train_delay_facts updated with weather columns')
print('    New columns: weather_code, precipitation_mm, weather_risk_score')

# ── Verify ────────────────────────────────────────────────────────────────────
spark.table(f'{CATALOG}.{DATABASE}.train_delay_facts') \
    .select('station', 'travel_date', 'delay_min',
            'weather_code', 'precipitation_mm', 'weather_risk_score') \
    .limit(10) \
    .display()