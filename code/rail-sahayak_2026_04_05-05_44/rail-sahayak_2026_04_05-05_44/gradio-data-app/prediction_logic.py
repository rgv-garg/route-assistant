"""
Prediction logic for Railway Delay Predictor App.
Uses local model file and pre-loaded station data (CSV).
"""
import os
import joblib
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# Configuration
MODEL_PATH = 'rail_delay_rf_model.pkl'
STATION_CSV = 'station_lookup.csv'  # Pre-exported station data
FEATURES = ['train_number', 'station_enc', 'month', 'day', 'day_of_week', 'week', 'day_type', 'rolling_72hr_avg_delay', 'is_maintenance_likely']

@dataclass
class StationDelay:
    """Predicted delay for a single station stop."""
    station: str
    predicted_delay_min: float
    delay_category: str

@dataclass
class TrainPrediction:
    """Full prediction result for a train."""
    train_number: int
    prediction_date: str
    stops: List[StationDelay]
    error: Optional[str] = None

# Load model once at module import
try:
    _saved = joblib.load(MODEL_PATH)
    _model = _saved['model']
    _label_encoder = _saved['label_encoder']
    print(f'✓ Model loaded from {MODEL_PATH}')
except Exception as e:
    print(f'⚠️ Model loading failed: {e}')
    _model = None
    _label_encoder = None

# Load station lookup once at module import
try:
    _station_lookup = pd.read_csv(STATION_CSV)
    print(f'✓ Station lookup loaded from {STATION_CSV} ({len(_station_lookup)} rows)')
except Exception as e:
    print(f'⚠️ Station lookup loading failed: {e}')
    _station_lookup = pd.DataFrame()

def _categorise_delay(delay_min: float) -> str:
    """Categorize delay for UI display."""
    if delay_min <= 15:
        return 'On Time'
    elif delay_min <= 60:
        return 'Slight Delay'
    else:
        return 'Significant Delay'

def predict_train_delay(train_number: int, date: str) -> TrainPrediction:
    """
    Predict delays for all stations of a given train for a specific date.
    
    Args:
        train_number: Train number (e.g., 12673)
        date: Date string in format 'YYYY-MM-DD'
    
    Returns:
        TrainPrediction with list of StationDelay stops or error message
    """
    if _model is None or _label_encoder is None:
        return TrainPrediction(
            train_number=train_number,
            prediction_date=date,
            stops=[],
            error='Model not loaded. Please check server logs.'
        )
    
    if _station_lookup.empty:
        return TrainPrediction(
            train_number=train_number,
            prediction_date=date,
            stops=[],
            error='Station data not available. Please contact support.'
        )
    
    today = pd.to_datetime(date)
    
    # Get stations for this train from pre-loaded data
    train_stations_df = _station_lookup[_station_lookup['train_number'] == train_number]
    
    if train_stations_df.empty:
        return TrainPrediction(
            train_number=train_number,
            prediction_date=str(today),
            stops=[],
            error=f'Train {train_number} not found. Please check the train number.'
        )
    
    train_stations = train_stations_df['station'].unique()
    
    # Encode stations
    try:
        station_enc = _label_encoder.transform(train_stations)
    except Exception as e:
        return TrainPrediction(
            train_number=train_number,
            prediction_date=str(today),
            stops=[],
            error=f'Station encoding failed: {str(e)}'
        )
    
    # Build input dataframe with all required features
    input_df = pd.DataFrame({
        'train_number': train_number,
        'station': train_stations,
        'station_enc': station_enc,
        'month': today.month,
        'day': today.day,
        'day_of_week': today.weekday(),
        'week': int(today.isocalendar()[1]),
        'day_type': 1 if today.weekday() >= 5 else 0,  # 1=weekend, 0=weekday
        'rolling_72hr_avg_delay': 15.0,  # Default placeholder (no historical data)
        'is_maintenance_likely': 0  # Default: no maintenance
    })
    
    # Predict
    input_df['predicted_delay_min'] = _model.predict(input_df[FEATURES]).round(1)
    
    # Build response
    stops = [
        StationDelay(
            station=row['station'],
            predicted_delay_min=row['predicted_delay_min'],
            delay_category=_categorise_delay(row['predicted_delay_min'])
        )
        for _, row in input_df.sort_values('station').iterrows()
    ]
    
    return TrainPrediction(
        train_number=train_number,
        prediction_date=str(today),
        stops=stops
    )
