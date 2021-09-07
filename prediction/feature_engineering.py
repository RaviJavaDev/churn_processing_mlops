import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from utils.data_management import DataManagement
from utils.feature_scaling import FeatureScaling


class FeatureEngineering:
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.feature_scaling = FeatureScaling()
        self.data_mgmt = DataManagement()
        self.smote = SMOTE()

    def pre_process_data(self):

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
        x = self.dataset

        # from dataset observed that churn column is imbalanced and due to this there
        # is higher chances our model will not predict better we can do up sampling using SMOTE from imblearn
        # x, y = self.smote.fit_resample(x, y)

        final_df = x
        final_df.to_csv('./processed_data/cleaned_data.csv', sep=',', index=None, header=True)

        return x

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

        self.dataset.drop(labels=cols, inplace=True, axis=1)

    def fill_total_charge(self, monthly_charges, tenure):
        if tenure == 0:
            total_charges = monthly_charges
        else:
            total_charges = tenure * monthly_charges
        return np.round(total_charges, 2)

    def process_categorical_features(self):

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
                       'TechSupport', 'StreamingMovies', 'PaperlessBilling', 'OnlineBackup', 'StreamingTV']

        for col in yes_no_cols:
            self.dataset[col].replace({'Yes': 1, 'No': 0}, inplace=True)
            self.dataset[col] = self.dataset[col].astype(dtype='int8')

            # convert InternetService,Contract,PaymentMethod columns into numeric using Label Encoding
            label_encode_cols = ['InternetService', 'Contract', 'PaymentMethod']
            label_encoder = LabelEncoder()
            for label_encode_col in label_encode_cols:
                self.dataset[label_encode_col] = label_encoder.fit_transform(self.dataset[label_encode_col])
                self.dataset[label_encode_col] = self.dataset[label_encode_col].astype(dtype='int8')

        return self.dataset


if __name__ == '__main__':
    df = pd.read_csv('../raw_data/customer_churn.csv')
    dataset = df.copy()
    feature_eng = FeatureEngineering(dataset)
    x_train, y_train, x_test, y_test = feature_eng.pre_process_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
