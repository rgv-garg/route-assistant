import os
from databricks import sql
from databricks.sdk.core import Config
import gradio as gr
import pandas as pd
import prediction_logic as pl

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

def sqlQuery(query: str) -> pd.DataFrame:
    """Execute SQL query and return results as pandas DataFrame."""
    cfg = Config() # Pull environment variables for auth
    
    # Fix: credentials_provider should return a HeaderFactory, not the authenticate method
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=cfg.authenticate  # Remove lambda wrapper
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall_arrow()
            if result is None:
                return pd.DataFrame()
            return result.to_pandas()

# Placeholder functions for predictors and route map
def delay_predictor(source, train_number, date_time):
    """Predict railway delays based on historical data."""
    try:
        # Convert train_number from string to int
        train_number = int(train_number)
        result = pl.predict_train_delay(train_number, date_time)
        if result is not None:
            if result.error:
                return f"Error: {result.error}"
            
            # Find the matching station
            delay = None
            for stop in result.stops:
                if stop.station == source:
                    delay = stop.predicted_delay_min
                    break
            
            if delay is not None:
                return f"Predicted Delay at {source}: {delay:.2f} minutes"
            else:
                return f"Station '{source}' not found in route for train {train_number}"
        return "Predicted Delay: N/A (no data found)"
    except Exception as e:
        return f"Error: {str(e)}"

def waiting_list_predictor(source, train_number):
    """Predict waiting list size based on historical data."""
    try:
        query = f"""SELECT avg(waiting_list) as avg_waiting 
                    FROM main.default.waiting_list 
                    WHERE source_station='{source}' AND train_number='{train_number}'"""
        result = sqlQuery(query)
        if not result.empty and 'avg_waiting' in result.columns:
            waiting = result['avg_waiting'].iloc[0]
            return f"Predicted Waiting List: {waiting:.0f}" if pd.notna(waiting) else "Predicted Waiting List: N/A"
        return "Predicted Waiting List: N/A (no data found)"
    except Exception as e:
        return f"Error: {str(e)}"

def route_map(source, destination, date_time):
    """Find routes between source and destination stations."""
    try:
        query = f"""SELECT * FROM main.default.schedule 
                    WHERE source_station='{source}' 
                    AND destination_station='{destination}' 
                    AND scheduled_departure='{date_time}'"""
        routes = sqlQuery(query)
        if routes.empty:
            return "No route found."
        route_str = "\n".join([
            f"Train {row['train_number']}: {row['source_station']} -> {row['destination_station']}" 
            for _, row in routes.iterrows()
        ])
        return f"Possible Routes:\n{route_str}"
    except Exception as e:
        return f"Error: {str(e)}"

# Skip database query at startup - use text inputs instead
station_list = []
print('Using text input for stations (no database query at startup)')

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    with gr.Tab("Delay Predictor"):
        gr.Markdown("# Railway Delay Predictor")
        with gr.Row():
            if station_list:
                delay_source = gr.Dropdown(choices=station_list, label="Source Station", interactive=True)
            else:
                delay_source = gr.Textbox(label="Source Station", placeholder="Enter station name", interactive=True)
            delay_train_number = gr.Textbox(label="Train Number", interactive=True)
            delay_date_time = gr.Textbox(label="Date and Time", placeholder="YYYY-MM-DD", interactive=True)
        delay_output = gr.Markdown()
        def predict_delay(src, tn, dt):
            return delay_predictor(src, tn, dt)
        gr.Button("Predict Delay").click(
            predict_delay,
            inputs=[delay_source, delay_train_number, delay_date_time],
            outputs=delay_output
        )

    with gr.Tab("Waiting List Predictor"):
        gr.Markdown("# Waiting List Predictor")
        if station_list:
            wl_source = gr.Dropdown(choices=station_list, label="Source Station", interactive=True)
        else:
            wl_source = gr.Textbox(label="Source Station", placeholder="Enter station name", interactive=True)
        wl_train_number = gr.Textbox(label="Train Number", interactive=True)
        wl_output = gr.Markdown()
        def predict_waiting(src, tn):
            return waiting_list_predictor(src, tn)
        gr.Button("Predict Waiting List").click(predict_waiting, inputs=[wl_source, wl_train_number], outputs=wl_output)

    with gr.Tab("Route Map"):
        gr.Markdown("# Route Map Finder")
        if station_list:
            rm_source = gr.Dropdown(choices=station_list, label="Source Station", interactive=True)
            rm_destination = gr.Dropdown(choices=station_list, label="Destination Station", interactive=True)
        else:
            rm_source = gr.Textbox(label="Source Station", placeholder="Enter source station", interactive=True)
            rm_destination = gr.Textbox(label="Destination Station", placeholder="Enter destination station", interactive=True)
        rm_date_time = gr.Textbox(label="Date and Time", placeholder="YYYY-MM-DD", interactive=True)
        rm_output = gr.Markdown()
        def find_route(src, dst, dt):
            return route_map(src, dst, dt)
        gr.Button("Find Route").click(find_route, inputs=[rm_source, rm_destination, rm_date_time], outputs=rm_output)

if __name__ == "__main__":
    demo.launch()
