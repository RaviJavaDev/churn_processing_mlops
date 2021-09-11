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
        try:
            self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                               y_train=y_train,
                                                                                               estimator=self.lbm_classifier,
                                                                                               params=params)
            self.lbm_classifier = self.best_estimator
            self.lbm_classifier.fit(x_train, y_train)
        except Exception as e:
            self.logger.error(f'error in LightGBM fit e: {e}')
            raise e
        self.logger.info('***** LightGBM Model building Finished *****')

    def predict(self, x_test):
        self.logger.info('***** In LightGBM Predict Started *****')
        try:
            y_pred = self.lbm_classifier.predict(x_test)
        except Exception as e:
            self.logger.error(f'error in LightGBM predict e: {e}')
            raise e
        self.logger.info('***** In LightGBM Predict Finished *****')
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        self.logger.info('***** In LightGBM evaluate_performance Started *****')
        try:
            score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
            tpr, thresholds = self.evaluation_metric.evaluate_performance(
                self.lbm_classifier, x, y_test, y_pred)
        except Exception as e:
            self.logger.error(f'error in LightGBM evaluate_performance e: {e}')
            raise e
        self.logger.info('***** In LightGBM evaluate_performance Finished *****')
        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
               tpr, thresholds

    def get_params(self):
        self.logger.info('***** In LightGBM get_params *****')
        return self.best_params
