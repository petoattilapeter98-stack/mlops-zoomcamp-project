# Conda
conda info --envs
conda activate my-exp-tr

# Jupyter Notebook
conda activate my-exp-tr
jupyter notebook

# MLflow Server
cd 02-experiment-tracking
conda activate my-exp-tr
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local