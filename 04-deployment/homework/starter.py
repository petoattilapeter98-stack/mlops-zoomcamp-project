import pickle
import pandas as pd
import numpy as np

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def apply_model():
    categorical = ['PULocationID', 'DOLocationID']
    year = 2023
    month = 3
    
    dv, model = load_model()
    df = read_data(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet", categorical)
    
    # Prepare features and predict durations
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    # Create ride_id column (like in the course videos)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    # Build the results dataframe and save to parquet
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    output_file = 'predictions.parquet'
    df_result.to_parquet(
     output_file,
     engine='pyarrow',
     compression=None,
     index=False
    )
    print(f"Saved predictions to {output_file}")
    print(f"Standard deviation of predicted duration: {np.std(y_pred)}")

if __name__ == "__main__":
    apply_model()