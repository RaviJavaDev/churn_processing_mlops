from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureScaling:
    def __init__(self, logger):
        self.logger = logger
        self.min_max_scalar = MinMaxScaler()
        self.standard_scalar = StandardScaler()

    def min_max_scaling(self, dataset, columns):
        """

        """
        self.logger.info('***** In FeatureScaling min_max_scaling Started *****')
        try:
            dataset[columns] = self.min_max_scalar.fit_transform(dataset[columns])
        except Exception as e:
            self.logger.error(f'error in FeatureScaling min_max_scaling e: {e}')
            raise e
        self.logger.info('***** In FeatureScaling min_max_scaling Finished *****')
        return dataset

    def standard_scaling(self, dataset, columns):
        """
        """
        self.logger.info('***** In FeatureScaling standard_scaling Started *****')
        try:
            dataset[columns] = self.standard_scalar.fit_transform(dataset[columns])
        except Exception as e:
            self.logger.error(f'error in FeatureScaling standard_scaling e: {e}')
            raise e
        self.logger.info('***** In FeatureScaling standard_scaling Finished *****')
        return dataset
