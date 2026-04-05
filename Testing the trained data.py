# Imports & Config 
import joblib
import datetime
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from pyspark.sql import functions as F

#  Constants
MODEL_PATH = '/Volumes/workspace/default/raildelaydata/rail_delay_rf_model.pkl'
CATALOG    = 'workspace'
DATABASE   = 'default'
TABLE      = 'train_delay_facts'
FEATURES = [
    'train_number',
    'station_enc',
    'month',
    'day',
    'day_of_week',
    'week',
    'day_type',
    'rolling_72hr_avg_delay',
    'is_maintenance_likely',
    'weather_code',              # ← new
    'precipitation_mm',          # ← new
    'weather_risk_score',        # ← new
]

print('Config loaded')

#  Data classes — clean contract for UI integration 
@dataclass
class StationDelay:
    """Predicted delay for a single station stop."""
    station:             str
    predicted_delay_min: float
    delay_category:      str    # 'On Time' / 'Slight Delay' / 'Significant Delay'

@dataclass
class TrainPrediction:
    """Full prediction result for a train — returned to the UI."""
    train_number:  int
    prediction_date: str
    stops:         List[StationDelay]
    error:         Optional[str] = None   # populated if something went wrong

print(' Data classes ready')
# Load model & station lookup ONCE at startup 
# These stay in memory — subsequent calls are instant

def _load_model(model_path: str) -> dict:
    """Load saved Random Forest model and LabelEncoder from DBFS."""
    try:
        saved = joblib.load(model_path)
        print(f' Model loaded from {model_path}')
        return saved
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Model not found at {model_path}. '
            f'Please run Notebook 1 to train and save the model first.'
        )

def _load_station_lookup(catalog: str, database: str, table: str) -> pd.DataFrame:
    """Load distinct train→station mapping + rolling avg + latest weather."""
    df = (
        spark.table(f'{catalog}.{database}.{table}')
        .select(
            'train_number', 'station',
            'rolling_72hr_avg_delay', 'is_maintenance_likely',
            'weather_code', 'precipitation_mm', 'weather_risk_score'  # ← new
        )
        .groupBy('train_number', 'station')
        .agg(
            F.avg('rolling_72hr_avg_delay').alias('rolling_72hr_avg_delay'),
            F.max('is_maintenance_likely').alias('is_maintenance_likely'),
            F.avg('weather_code').alias('weather_code'),              # ← new
            F.avg('precipitation_mm').alias('precipitation_mm'),      # ← new
            F.avg('weather_risk_score').alias('weather_risk_score'),  # ← new
        )
        .toPandas()
    )
    print(f'✅  Station lookup loaded — {df["train_number"].nunique()} trains, {df["station"].nunique()} unique stations')
    return df
# Load once
_saved         = _load_model(MODEL_PATH)
_model         = _saved['model']
_label_encoder = _saved['label_encoder']
_station_lookup = _load_station_lookup(CATALOG, DATABASE, TABLE)
# Helper: categorise delay for UI badge/colour 
def _categorise_delay(delay_min: float) -> str:
    """
    Maps predicted delay minutes to a human-readable category.
    Matches the legend used in the running history UI:
      Green  → On Time          (0–15 min)
      Orange → Slight Delay     (15–60 min)
      Red    → Significant Delay(>60 min)
    """
    if delay_min <= 15:
        return 'On Time'
    elif delay_min <= 60:
        return 'Slight Delay'
    else:
        return 'Significant Delay'


# Main prediction function — call this from UI 
def predict_train_delays(train_number: int, date: str) -> TrainPrediction:
    """
    Predict delays for all stations of a given train for today.

    Args:
        train_number: The train number entered by the user (e.g. 12673)

    Returns:
        TrainPrediction dataclass with a list of StationDelay stops.
        On error, returns TrainPrediction with error message populated.

    Example (UI integration):
        result = predict_train_delays(12673)
        if result.error:
            show_error(result.error)
        else:
            for stop in result.stops:
                render_stop(stop.station, stop.predicted_delay_min, stop.delay_category)
    """
    today = pd.to_datetime(date)

    # Validate train number
    train_stations = (
        _station_lookup[_station_lookup['train_number'] == train_number]['station']
        .unique()
    )

    if len(train_stations) == 0:
        return TrainPrediction(
            train_number    = train_number,
            prediction_date = str(today),
            stops           = [],
            error           = f'Train {train_number} not found. Please check the train number and try again.'
        )

    #  Build input for model
    try:
        station_enc = _label_encoder.transform(train_stations)
    except Exception:
        return TrainPrediction(
            train_number    = train_number,
            prediction_date = str(today),
            stops           = [],
            error           = f'Station encoding failed for train {train_number}.'
        )

    # Indian holidays
    INDIAN_HOLIDAYS = {
        '2025-01-26', '2025-03-17', '2025-04-14', '2025-04-18',
        '2025-08-15', '2025-10-02', '2025-10-20', '2025-11-05', '2025-12-25'
    }

    # Compute day_type for the query date
    date_str = today.strftime('%Y-%m-%d')
    if date_str in INDIAN_HOLIDAYS:
        day_type = 2   # holiday
    elif today.weekday() >= 5:
        day_type = 1   # weekend
    else:
        day_type = 0   # weekday

    # Get latest rolling avg + maintenance flag per station from lookup
    station_meta = (
        _station_lookup[_station_lookup['train_number'] == train_number]
        .set_index('station')
    )

    input_df = pd.DataFrame({
        'train_number':           train_number,
        'station':                train_stations,
        'station_enc':            station_enc,
        'month':                  today.month,
        'day':                    today.day,
        'day_of_week':            today.weekday(),
        'week':                   int(today.isocalendar()[1]),
        'day_type':               day_type,
        'rolling_72hr_avg_delay': [
            station_meta.loc[s, 'rolling_72hr_avg_delay']
            if s in station_meta.index else 0.0
            for s in train_stations
        ],
        'is_maintenance_likely': [
            station_meta.loc[s, 'is_maintenance_likely']
            if s in station_meta.index else 0
            for s in train_stations
        ],
        # ── New weather features ──────────────────────────────────────────────
        'weather_code': [
            station_meta.loc[s, 'weather_code']
            if s in station_meta.index else 0.0
            for s in train_stations
        ],
        'precipitation_mm': [
            station_meta.loc[s, 'precipitation_mm']
            if s in station_meta.index else 0.0
            for s in train_stations
        ],
        'weather_risk_score': [
            station_meta.loc[s, 'weather_risk_score']
            if s in station_meta.index else 0.0
            for s in train_stations
        ],
    })

    # Run prediction 
    input_df['predicted_delay_min'] = _model.predict(input_df[FEATURES]).round(1)

    #  Build response
    stops = [
        StationDelay(
            station             = row['station'],
            predicted_delay_min = row['predicted_delay_min'],
            delay_category      = _categorise_delay(row['predicted_delay_min'])
        )
        for _, row in input_df.sort_values('station').iterrows()
    ]

    return TrainPrediction(
        train_number    = train_number,
        prediction_date = str(today),
        stops           = stops
    )

print('predict_train_delays() function ready')

#  Run prediction — only line you change
QUERY_TRAIN = 12673    # ← enter any train number here
date = "2026-04-02"
result = predict_train_delays(QUERY_TRAIN, date)

if result.error:
    print(f' Error: {result.error}')
else:
    print(f' Train {result.train_number} — Predicted delays for {result.prediction_date}')
    print(f'{"-"*45}')
    print(f'{"Station":<10} {"Delay (min)":>12} {"Status"}')
    print(f'{"-"*45}')
    for stop in result.stops:
        print(f'{stop.station:<10} {stop.predicted_delay_min:>12} {stop.delay_category}')
    print(f'{"-"*45}')

    # Also display as a clean table
    display(pd.DataFrame([{
        'Station':              s.station,
        'Predicted Delay (min)': s.predicted_delay_min,
        'Status':               s.delay_category
    } for s in result.stops]))

    import json
from dataclasses import asdict

result_json = json.dumps(asdict(result), indent=2)
print(result_json)