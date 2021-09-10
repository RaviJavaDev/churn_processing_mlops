import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from utils.data_management import DataManagement
from utils.feature_scaling import FeatureScaling
from sklearn.preprocessing import LabelEncoder


class FeatureEngineering:
    def __init__(self, dataset, config, logger):
        self.config = config
        self.dataset = dataset
        self.logger = logger
        self.feature_scaling = FeatureScaling(self.logger)
        self.data_mgmt = DataManagement(self.logger)
        self.smote = SMOTE()

    def pre_process_data(self):
        self.logger.info('***** Training Feature Engineering Pipeline pre_process_data Started *****')
        # remove unwanted cols
        self.remove_cols(['customerID'])

        # convert column TotalCharges into numeric
        self.dataset['TotalCharges'] = pd.to_numeric(self.dataset['TotalCharges'], errors='coerce')

        # We will follow approach derive the value of TotalCharges for null using Monthly charge & Tenure
        self.dataset['TotalCharges'] = self.dataset.apply(
            lambda x: x['TotalCharges'] if not pd.isna(x['TotalCharges']) else self.fill_total_charge(
                x['MonthlyCharges'], x['tenure']), axis=1)

        # process categorical features
        self.dataset = self.process_categorical_features()

        # scale down the numeric feature using MinMaxScalar
        numeric_features = list(self.dataset.columns[self.dataset.dtypes == 'float64'])
        numeric_features.append('tenure')
        self.dataset = self.feature_scaling.min_max_scaling(self.dataset, numeric_features)

        # divide independent and dependent columns
        x = self.dataset.drop('Churn', axis=1)
        y = self.dataset['Churn']

        # from dataset observed that churn column is imbalanced and due to this there
        # is higher chances our model will not predict better we can do up sampling using SMOTE from imblearn
        x, y = self.smote.fit_resample(x, y)

        final_df = pd.concat([x, y], axis=1)
        final_df.to_csv('./data/cleaned_data/training/cleaned_data.csv', sep=',', index=None, header=True)

        self.logger.info('***** Training Feature Engineering Pipeline  pre_process_data Finished *****')

        return x, y

    def remove_cols(self, cols):
        """ removes columns from dataframe.

        Parameters
        ----------
            cols : array
                List of columns to be removed.

        Returns
        -------
            df:
                dataframe without cols.
        """
        self.logger.info(f'***** In  remove_cols removing column {cols} Started *****')
        self.dataset.drop(labels=cols, inplace=True, axis=1)
        self.logger.info(f'***** In  remove_cols removing column {cols} Finished *****')

    def fill_total_charge(self, monthly_charges, tenure):
        self.logger.info('***** In  fill_total_charge Started *****')
        if tenure == 0:
            total_charges = monthly_charges
        else:
            total_charges = tenure * monthly_charges
        self.logger.info('***** In  fill_total_charge Finished *****')
        return np.round(total_charges, 2)

    def process_categorical_features(self):
        self.logger.info('***** In  process_categorical_features started *****')
        # for gender column replace with value 1 & 0
        self.dataset['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)

        # for column MultipleLines  replace value No phone service with No
        self.dataset['MultipleLines'].replace('No phone service', 'No', inplace=True)

        # for column OnlineSecurity  replace No internet service with No
        self.dataset['OnlineSecurity'].replace('No internet service', 'No', inplace=True)

        # for column DeviceProtection  replace No internet service with No
        self.dataset['DeviceProtection'].replace('No internet service', 'No', inplace=True)

        # for column TechSupport replace No internet service with No
        self.dataset['TechSupport'].replace('No internet service', 'No', inplace=True)

        # for column StreamingMovies replace No internet service with No
        self.dataset['StreamingMovies'].replace('No internet service', 'No', inplace=True)

        # for column OnlineBackup replace No internet service with No
        self.dataset['OnlineBackup'].replace('No internet service', 'No', inplace=True)

        # for column StreamingTV replace No internet service with No
        self.dataset['StreamingTV'].replace('No internet service', 'No', inplace=True)

        # for columns replace Yes with 1 and No 0
        yes_no_cols = ['Partner', 'PhoneService', 'Dependents', 'MultipleLines', 'OnlineSecurity', 'DeviceProtection',
                       'TechSupport', 'StreamingMovies', 'PaperlessBilling', 'OnlineBackup', 'StreamingTV', 'Churn']

        for col in yes_no_cols:
            self.dataset[col].replace({'Yes': 1, 'No': 0}, inplace=True)
            self.dataset[col] = self.dataset[col].astype(dtype='int8')

        # convert InternetService,Contract,PaymentMethod columns into numeric using Label Encoding
        label_encode_cols = ['InternetService', 'Contract', 'PaymentMethod']
        label_encoder = LabelEncoder()
        for label_encode_col in label_encode_cols:
            self.dataset[label_encode_col] = label_encoder.fit_transform(self.dataset[label_encode_col])
            self.dataset[label_encode_col] = self.dataset[label_encode_col].astype(dtype='int8')

        self.logger.info('***** In  process_categorical_features finished *****')

        return self.dataset


if __name__ == '__main__':
    df = pd.read_csv('../../data/raw_data/customer_churn.csv')
    dataset = df.copy()
    feature_eng = FeatureEngineering(dataset)
    x_train, y_train, x_test, y_test = feature_eng.pre_process_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
