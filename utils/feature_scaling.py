from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureScaling:
    def __init__(self):
        self.min_max_scalar = MinMaxScaler()
        self.standard_scalar = StandardScaler()

    def min_max_scaling(self, dataset,columns):
        """

        """
        dataset[columns] = self.min_max_scalar.fit_transform(dataset[columns])
        return dataset

    def standard_scaling(self, dataset, columns):
        """
        """
        dataset[columns] = self.standard_scalar.fit_transform(dataset[columns])
        return dataset
