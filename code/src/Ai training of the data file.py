import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pyspark.sql import functions as F
import datetime

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG    = 'workspace'
DATABASE   = 'default'
TABLE      = 'train_delay_facts'
MODEL_PATH = '/Volumes/workspace/default/raildelaydata/rail_delay_rf_model.pkl'


# ── Load from Delta Lake ──────────────────────────────────────────────────────
pdf = spark.table(f'{CATALOG}.{DATABASE}.{TABLE}').toPandas()
pdf['travel_date'] = pd.to_datetime(pdf['travel_date'])
pdf['month']       = pdf['travel_date'].dt.month
pdf['day']         = pdf['travel_date'].dt.day
pdf['day_of_week'] = pdf['travel_date'].dt.dayofweek
pdf['week']        = pdf['travel_date'].dt.isocalendar().week.astype(int)
pdf = pdf.dropna(subset=['delay_min'])

# Confirm all new columns loaded
print('New columns check:')
print(pdf[['day_type', 'rolling_72hr_avg_delay', 'is_maintenance_likely',
           'weather_code', 'precipitation_mm', 'weather_risk_score']].describe())
# ── Encode stations ───────────────────────────────────────────────────────────
le = LabelEncoder()
pdf['station_enc'] = le.fit_transform(pdf['station'])

FEATURES = [
    'train_number',
    'station_enc',
    'month',
    'day',
    'day_of_week',
    'week',
    'day_type',                  # from Notebook 0
    'rolling_72hr_avg_delay',    # from Notebook 0
    'is_maintenance_likely',     # from Notebook 0
    'weather_code',              # ← new
    'precipitation_mm',          # ← new
    'weather_risk_score',        # ← new
]
X = pdf[FEATURES]
y = pdf['delay_min']

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')

# ── Train Random Forest ───────────────────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print('✅  Model trained')

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = rf.predict(X_test)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f'RMSE : {rmse:.2f} min')
print(f'MAE  : {mae:.2f} min')
print(f'R²   : {r2:.4f}')

# ── Save model to DBFS ────────────────────────────────────────────────────────
joblib.dump({'model': rf, 'label_encoder': le}, MODEL_PATH)
print(f'✅  Model saved to {MODEL_PATH}')

# ── Predict next delay — just enter train number ──────────────────────────────
QUERY_TRAIN = 12673          # ← only this needs to change

# Automatically use today as the prediction date
today = datetime.date.today()
print(f'\n📅  Predicting for today: {today}')

train_stations = pdf[pdf['train_number'] == QUERY_TRAIN]['station'].unique()

if len(train_stations) == 0:
    raise ValueError(f'Train {QUERY_TRAIN} not found in dataset')

input_df = pd.DataFrame({
    'train_number': QUERY_TRAIN,
    'station':      train_stations,
    'station_enc':  le.transform(train_stations),
    'month':        today.month,
    'day':          today.day,
    'day_of_week':  today.weekday(),
    'week':         int(today.isocalendar()[1]),
    'day_type':     0 if today.weekday() < 5 else 1,  # 0=weekday, 1=weekend
    'rolling_72hr_avg_delay': 0,  # Default to 0 for prediction
    'is_maintenance_likely':  0,  # Default to 0 (no maintenance) for prediction
    'weather_code':          0,  # Default to 0 for prediction
    'precipitation_mm':      0,  # Default to 0 for prediction
    'weather_risk_score':    0,  # Default to 0 for prediction
})

input_df['predicted_delay_min'] = rf.predict(input_df[FEATURES]).round(1)

print(f'\n🚆 Train {QUERY_TRAIN} — Predicted delays for today ({today})')
display(
    input_df[['station', 'predicted_delay_min']]
    .sort_values('station')
    .reset_index(drop=True)
)

# ── Save predictions to Delta Lake ────────────────────────────────────────────
spark.createDataFrame(input_df[['station', 'predicted_delay_min']]) \
    .withColumn('train_number', F.lit(QUERY_TRAIN)) \
    .withColumn('query_date',   F.lit(str(today))) \
    .write.format('delta').mode('overwrite') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(f'{CATALOG}.{DATABASE}.delay_predictions')

print(f'✅  Predictions saved to {CATALOG}.{DATABASE}.delay_predictions')

