from training import Training
import pandas as pd


def train():
    train_model = Training()
    df = pd.read_csv('./raw_data/customer_churn.csv')
    train_model.train(df)


if __name__ == '__main__':
    train()
