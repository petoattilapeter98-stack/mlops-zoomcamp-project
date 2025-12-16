# Conda
conda info --envs
conda activate my-exp-tr

# Jupyter Notebook
conda activate my-exp-tr
jupyter notebook

# 02 - MLflow Server
cd 02-experiment-tracking
conda activate my-exp-tr
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local

# 03 - Orchestration with MLflow and Prefect
cd 03-orchestration
conda activate my-exp-tr
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local

cd 03-orchestration
conda activate my-exp-tr
prefect server start