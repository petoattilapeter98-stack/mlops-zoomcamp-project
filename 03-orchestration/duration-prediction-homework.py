#!/usr/bin/env python

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta, datetime

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("03-ho-nyc-taxi-experiment")
mlflow.sklearn.autolog()

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)



@task(
    retries=2,
    retry_delay_seconds=60,
    timeout_seconds=600,
    tags=["data-loading"]
)
def read_dataframe(year, month):
    logger = get_run_logger()
    logger.info(f"Reading data for {year}-{month:02d}")
    
    try:
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)
        logger.info(f"Downloaded {len(df)} records")

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

        df = df[(df.duration >= 1) & (df.duration <= 60)]
        logger.info(f"Filtered to {len(df)} records after duration filtering")

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)

        return df
    except Exception as e:
        logger.error(f"Failed to read dataframe: {str(e)}")
        raise


@task(
    tags=["feature-engineering"],
    timeout_seconds=300
)
def create_X(df, dv=None):
    logger = get_run_logger()
    logger.info(f"Creating feature matrix from {len(df)} records")
    
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
        logger.info(f"Fitted DictVectorizer with {X.shape[1]} features")
    else:
        X = dv.transform(dicts)
        logger.info(f"Transformed data to {X.shape[1]} features")

    return X, dv


@task(
    tags=["model-training"],
    timeout_seconds=1800
)
def train_model(X_train, y_train, X_val, y_val, dv):
    logger = get_run_logger()
    logger.info(f"Training LinearRegression with {X_train.shape[0]} training samples and {X_val.shape[0]} validation samples")

    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        logger.info(f"Intercept: {lr.intercept_}")
        mlflow.log_metric("intercept", float(lr.intercept_))

        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        logger.info(f"LinearRegression RMSE: {rmse:.4f}")

        # save the preprocessor
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        logger.info(f"Saved preprocessor artifact")

        # log sklearn model
        mlflow.sklearn.log_model(lr, artifact_path="model_sklearn")
        logger.info(f"Logged LinearRegression model to MLflow")

        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.register_model( model_uri=f"runs:/{run_id}/model", name = "LinearRegressionModel")

        return run_id


@task(
    tags=["artifacts"],
    timeout_seconds=60
)
def save_run_id(run_id):
    logger = get_run_logger()
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    logger.info(f"Saved run_id to file: {run_id}")


@flow(
    name="taxi-trip-duration-training",
    description="Train XGBoost model for NYC taxi trip duration prediction",
    log_prints=True
)
def taxi_training_flow(year: int = None, month: int = None):
    logger = get_run_logger()
    
    # If year/month not provided, use previous month
    if year is None or month is None:
        today = datetime.now()
        if today.month == 1:
            year = today.year - 1
            month = 12
        else:
            year = today.year
            month = today.month - 1
    
    logger.info(f"Starting taxi training flow for {year}-{month:02d}")
    
    # Load data
    df = read_dataframe(year=year, month=month)
    
    # Split into train/validation (80/20 split)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]
    logger.info(f"Split data: {len(df_train)} training samples, {len(df_val)} validation samples")

    # Feature engineering
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    # Prepare targets
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    # Train model
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    
    # Save artifacts
    save_run_id(run_id)
    
    logger.info(f"Completed taxi training flow successfully")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = taxi_training_flow(year=args.year, month=args.month)
    print(f"MLflow run_id: {run_id}")

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path="DownloadedModel")
