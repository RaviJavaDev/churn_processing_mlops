import yaml
import argparse


class ReadParams:
    def __init__(self):
        pass

    def read(self, config_path):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config


if __name__ == '__main__':
    read_params = ReadParams()
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='../params.yaml')
    parsed_args = args.parse_args()
    config = read_params.read(parsed_args.config)
    print(config['estimators']['LogisticRegression']['params'])
