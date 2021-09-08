from training_service import Training
import pandas as pd
import flask


def train():
    train_model = Training()
    df = pd.read_csv('data/raw_data/customer_churn.csv')
    train_model.train(df)


if __name__ == '__main__':
    train()
