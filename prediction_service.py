import argparse

import joblib
import mlflow

from mlflow.tracking import MlflowClient

from prediction_service.pre_process_data.feature_engineering import FeatureEngineering
from utils.read_params import ReadParams

import os

import pandas as pd


class Prediction:
    def __init__(self, config, data):
        self.config = config
        self.feature_engineering = FeatureEngineering(data, config)

    def select_best_model(self):
        mlflow_config = self.config['mlflow_config']
        mlflow.set_tracking_uri(mlflow_config['remote_server_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])
        model_name = mlflow_config["registered_model_name"]
        runs = mlflow.search_runs(experiment_ids=1, order_by=['metrics.accuracy DESC'])
        highest_accuracy_run = runs[runs['metrics.accuracy'] == max(runs['metrics.accuracy'])]
        run_id = highest_accuracy_run['run_id'][0]
        mlflow_client = MlflowClient()
        filter_string = "run_id='{}'".format(run_id)
        artifact_list = mlflow_client.search_model_versions(filter_string)
        model_path = os.path.join(artifact_list[0].source, 'model.pkl')
        return model_path

    def predict(self):
        model_path = self.select_best_model()
        model = joblib.load(model_path)
        x = self.feature_engineering.pre_process_data()
        result = model.predict(x).tolist()
        print(result)


if __name__ == '__main__':
    read_params = ReadParams()
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml')
    parsed_args = args.parse_args()
    config = read_params.read(parsed_args.config)

    dataset_path = config['raw_data']['raw_dataset']
    data = pd.read_csv(dataset_path)
    data = data.drop('Churn',axis=1).sample(5)
    print(data.shape)
    prediction = Prediction(config, data)

    prediction.predict()
