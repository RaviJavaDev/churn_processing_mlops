from sklearn.ensemble import RandomForestClassifier

from utils.evaluation_metrics import EvaluationMetrics
from utils.hyper_parameters_tuning import HyperParametersTuning


class RandomForest:
    def __init__(self, logger):
        self.logger = logger
        self.random_forest = RandomForestClassifier()
        self.hyper_param_tuning = HyperParametersTuning(self.logger)
        self.evaluation_metric = EvaluationMetrics(self.logger)
        self.best_estimator = None
        self.best_params = None

    def fit(self, x_train, y_train, params):
        self.logger.info('***** RandomForest Model building Started *****')
        self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                           y_train=y_train,
                                                                                           estimator=self.random_forest,
                                                                                           params=params)
        self.random_forest = self.best_estimator

        self.random_forest.fit(x_train, y_train)
        self.logger.info('***** RandomForest Model building Finished *****')

    def predict(self, x_test):
        self.logger.info('***** In RandomForest Predict Started *****')
        y_pred = self.random_forest.predict(x_test)
        self.logger.info('***** In RandomForest Predict Finished *****')
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        self.logger.info('***** In RandomForest evaluate_performance Started *****')
        score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
        tpr, thresholds = self.evaluation_metric.evaluate_performance(
            self.random_forest, x, y_test, y_pred)
        self.logger.info('***** In RandomForest evaluate_performance Finished *****')
        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
               tpr, thresholds

    def get_params(self):
        return self.best_params
