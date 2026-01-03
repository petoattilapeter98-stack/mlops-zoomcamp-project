import os
from datetime import datetime
import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def make_prepared_df():
    data = [
        (None, None, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]

    columns = [
        'PULocationID',
        'DOLocationID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'duration',
    ]

    categorical = ['PULocationID', 'DOLocationID']

    df = pd.DataFrame(data, columns=columns)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def main():
    year = 2023
    month = 1

    df_input = make_prepared_df()

    input_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"

    s3_endpoint = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint:
        options = {'client_kwargs': {'endpoint_url': s3_endpoint}}
    else:
        options = {}

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options,
    )


if __name__ == '__main__':
    main()
