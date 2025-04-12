# Placeholder for data_loader module
import pandas as pd

def load_data():
    # Example function to load data
    url = 'Telcom-Customer-Churn.csv'
    df = pd.read_csv(url)

    # Ensure consistent data types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                pass

    df.fillna(0, inplace=True)  # Replace NaN values with 0
    return df
