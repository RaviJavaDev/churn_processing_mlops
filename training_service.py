import argparse

import pandas as pd

from logger.file_logger import FileLogger
from training.model_mgmt.model_selection import ModelSelection
from training.pre_process_data.feature_engineering import FeatureEngineering
from utils.data_management import DataManagement
from utils.read_params import ReadParams


class Training:

    def __init__(self, data, config, log_path):
        self.data = data
        self.config = config
        self.logger = FileLogger(log_path)
        self.feature_engineering = FeatureEngineering(data, config, self.logger)
        self.model_selection = ModelSelection(config, self.logger)
        self.data_mgmt = DataManagement(self.logger)

    def train(self):
        """ Method to train the dataset on ML algorithm and save model_mgmt.

        """
        self.logger.info('***** Started Training *****')
        x, y = self.feature_engineering.pre_process_data()

        # divide data into train and test
        x_train, y_train, x_test, y_test = self.data_mgmt.train_test_split(x, y)

        self.model_selection.fit(x_train, y_train, x_test, y_test)
        self.logger.info('***** Finished Training *****')


if __name__ == '__main__':
    read_params = ReadParams()
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml')
    parsed_args = args.parse_args()
    config = read_params.read(parsed_args.config)

    dataset_path = config['raw_data']['raw_dataset']
    training_log_path = config['log_path']['training']
    df = pd.read_csv(dataset_path)
    training = Training(df, config,training_log_path)
    training.train()
