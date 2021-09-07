from sklearn.model_selection import train_test_split


class DataManagement:
    """ This class splits dataset into train set,test set and validation set.

    Author: Ravi Sonawane

    Date: 05-09-2021

    Email: ravisonawane20@gmail.com
    """

    def __init__(self):
        pass

    def train_test_split(self, *arrays, test_size=0.25, train_size=None, random_state=42, stratify=None):
        """ splits dataset into train and test set.

        Parameters
        ----------
            *arrays: list,numpy arrays,pandas dataframe
                data which needs to split.

            test_size: float or int, default=0.25
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split.

            train_size : float or int, default=None
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the train split.

            random_state: int
                Controls the shuffling applied to the data before applying the split.

            stratify:
                array-like, default=None

        Returns
        -------
            splitting : list, length=2 * len(arrays)
                List containing train-test split of inputs.

        """
        x_train, x_test, y_train, y_test = train_test_split(*arrays, test_size=test_size, train_size=train_size,
                                                            random_state=random_state, stratify=stratify)

        return x_train, y_train, x_test, y_test

    def valid_split(self, *arrays, val_size=0.10, random_state=42, stratify=None):
        """ splits dataset into train and validation set.

               Parameters
               ----------
                   *arrays: list,numpy arrays,pandas dataframe
                       data which needs to split.

                   val_size: float or int, default=0.25
                       If float, should be between 0.0 and 1.0 and represent the proportion
                       of the dataset to include in the test split.

                   random_state: int
                       Controls the shuffling applied to the data before applying the split.

                   stratify:
                       array-like, default=None

               Returns
               -------
                   splitting : list, length=2 * len(arrays)
                       List containing train-test split of inputs.

               """
        x_train, x_valid, y_train, y_valid = train_test_split(*arrays, test_size=val_size, random_state=random_state,
                                                              stratify=stratify)
        return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    data_mgmt = DataManagement()
    data_mgmt.train_test_split()
