from lightgbm import LGBMClassifier

from utils.evaluation_metrics import EvaluationMetrics
from utils.hyper_parameters_tuning import HyperParametersTuning


class LightGBM:
    def __init__(self, logger):
        self.logger = logger
        self.lbm_classifier = LGBMClassifier()
        self.hyper_param_tuning = HyperParametersTuning(self.logger)
        self.evaluation_metric = EvaluationMetrics(self.logger)
        self.best_estimator = None
        self.best_params = None

    def fit(self, x_train, y_train, params):
        self.logger.info('***** LightGBM Model building Started *****')
        self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                           y_train=y_train,
                                                                                           estimator=self.lbm_classifier,
                                                                                           params=params)
        self.lbm_classifier = self.best_estimator

        self.lbm_classifier.fit(x_train, y_train)
        self.logger.info('***** LightGBM Model building Finished *****')

    def predict(self, x_test):
        self.logger.info('***** In LightGBM Predict Started *****')
        y_pred = self.lbm_classifier.predict(x_test)
        self.logger.info('***** In LightGBM Predict Finished *****')
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        self.logger.info('***** In LightGBM evaluate_performance Started *****')
        score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
        tpr, thresholds = self.evaluation_metric.evaluate_performance(
            self.lbm_classifier, x, y_test, y_pred)
        self.logger.info('***** In LightGBM evaluate_performance Finished *****')
        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
               tpr, thresholds

    def get_params(self):
        self.logger.info('***** In LightGBM get_params *****')
        return self.best_params
