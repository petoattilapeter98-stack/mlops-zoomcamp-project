#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd

def my_def():
    return 1

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def read_data(filename):
    s3_endpoint = os.getenv('S3_ENDPOINT_URL')

    if s3_endpoint:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint
            }
        }
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)

    return df

def save_data(df, filename):
        s3_endpoint = os.getenv('S3_ENDPOINT_URL')

        if s3_endpoint:
            options = {
                'client_kwargs': {
                    'endpoint_url': s3_endpoint
                }
            }
            df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
        else:
            df.to_parquet(filename, engine='pyarrow', index=False)

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    #input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    print(f'Reading data from {input_file}')
    print(f'Writing results to {output_file}')

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file)
    df_transformed = prepare_data(df, categorical)
    df_transformed['ride_id'] = f'{year:04d}/{month:02d}_' + df_transformed.index.astype('str')


    dicts = df_transformed[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df_transformed['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)

if __name__ == '__main__':
    main(year=sys.argv[1], month=sys.argv[2])