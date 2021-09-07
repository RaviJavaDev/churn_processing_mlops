from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class HyperParametersTuning:
    def __init__(self):
        pass

    def get_best_estimator(self, x_train, y_train, estimator, params, n_iter=5, cv=5, n_jobs=-1):
        random_search_cv = RandomizedSearchCV(estimator=estimator, param_distributions=params, n_iter=n_iter, cv=cv,
                                              n_jobs=n_jobs)
        random_search_cv.fit(x_train, y_train)

        best_estimator = random_search_cv.best_estimator_

        best_params = random_search_cv.best_params_

        print('Random Search Best Estimator :', best_estimator)

        print('best params: ', best_estimator)

        return best_estimator, best_params
