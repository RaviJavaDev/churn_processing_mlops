from sklearn.linear_model import LogisticRegression

from utils.hyper_parameters_tuning import HyperParametersTuning
from utils.evaluation_metrics import EvaluationMetrics


class LogisticReg:

    def __init__(self, logger):
        self.logger = logger
        self.logistic_regression = LogisticRegression(self.logger)
        self.hyper_param_tuning = HyperParametersTuning(self.logger)
        self.evaluation_metric = EvaluationMetrics(self.logger)
        self.best_estimator = None
        self.best_params = None

    def fit(self, x_train, y_train, params):
        self.logger.info('***** LogisticReg Model building Started *****')
        try:
            self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                               y_train=y_train,
                                                                                               estimator=self.logistic_regression,
                                                                                               params=params)
            self.logistic_regression = self.best_estimator

            self.logistic_regression.fit(x_train, y_train)
        except Exception as e:
            self.logger.error(f'error in LogisticReg fit e: {e}')
            raise e
        self.logger.info('***** LogisticReg Model building Finished *****')

    def predict(self, x_test):
        self.logger.info('***** In LogisticReg Predict Started *****')
        try:
            y_pred = self.logistic_regression.predict(x_test)
        except Exception as e:
            self.logger.error(f'error in LogisticReg predict e: {e}')
            raise e
        self.logger.info('***** In LogisticReg Predict Finished *****')
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        self.logger.info('***** In LogisticReg evaluate_performance Started *****')
        try:
            score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
            tpr, thresholds = self.evaluation_metric.evaluate_performance(
                self.logistic_regression, x, y_test, y_pred)
        except Exception as e:
            self.logger.error(f'error in LogisticReg evaluate_performance e: {e}')
            raise e
        self.logger.info('***** In LogisticReg evaluate_performance Finished *****')
        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds

    def get_params(self):
        self.logger.info('***** In LogisticReg get_params *****')
        return self.best_params
