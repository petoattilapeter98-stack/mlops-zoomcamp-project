#!/usr/bin/env python

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task(
    retries=2,
    retry_delay_seconds=60,
    timeout_seconds=600,
    tags=["data-loading"],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24)
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

        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

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
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

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
    logger.info(f"Training model with {X_train.shape[0]} training samples and {X_val.shape[0]} validation samples")
    
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)
        logger.info(f"Logged parameters to MLflow")

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        logger.info(f"Model RMSE: {rmse:.4f}")

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        logger.info(f"Saved preprocessor artifact")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        logger.info(f"Logged XGBoost model to MLflow")

        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")
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
def taxi_training_flow(year: int, month: int):
    logger = get_run_logger()
    logger.info(f"Starting taxi training flow for {year}-{month:02d}")
    
    # Load data
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

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