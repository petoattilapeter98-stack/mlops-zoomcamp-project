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

# 04 - Deployment
# conda create --name my_deployment-fixed python=3.10
# conda activate my_deployment-fixed

cd 04-deployment/homework
pip install -r ./requirements.txt
jupyter notebook
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local

# 05 - Monitoring
# conda create --name monitoring python=3.11
# conda activate monitoring

cd 05-monitoring/homework
# pip install -r ./requirements.txt
docker-compose up --build
jupyter notebook
evindently ui

# 06 - best-practises
cd 04-deployment/homework
pipenv install
pipenv install --dev pytest
