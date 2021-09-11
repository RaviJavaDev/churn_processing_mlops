from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class HyperParametersTuning:
    def __init__(self, logger):
        self.logger = logger

    def get_best_estimator(self, x_train, y_train, estimator, params, n_iter=5, cv=5, n_jobs=-1):
        self.logger.info('***** HyperParametersTuning get_best_estimator Started *****')
        try:
            random_search_cv = RandomizedSearchCV(estimator=estimator, param_distributions=params, n_iter=n_iter, cv=cv,
                                                  n_jobs=n_jobs)
            random_search_cv.fit(x_train, y_train)

            best_estimator = random_search_cv.best_estimator_

            best_params = random_search_cv.best_params_

            self.logger.info(f'***** Best estimator : {best_estimator} *****')
        except Exception as e:
            self.logger.error(f'error in HyperParametersTuning get_best_estimator e: {e}')
            raise e

        self.logger.info('***** HyperParametersTuning get_best_estimator Finished *****')

        return best_estimator, best_params
