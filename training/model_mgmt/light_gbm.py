from lightgbm import LGBMClassifier

from utils.evaluation_metrics import EvaluationMetrics
from utils.hyper_parameters_tuning import HyperParametersTuning


class LightGBM:
    def __init__(self):
        self.lbm_classifier = LGBMClassifier()
        self.hyper_param_tuning = HyperParametersTuning()
        self.evaluation_metric = EvaluationMetrics()

    def fit(self, x_train, y_train, params):
        self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                           y_train=y_train,
                                                                                           estimator=self.lbm_classifier,
                                                                                           params=params)
        self.lbm_classifier = self.best_estimator

        self.lbm_classifier.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.lbm_classifier.predict(x_test)
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
        tpr, thresholds = self.evaluation_metric.evaluate_performance(
            self.lbm_classifier, x, y_test, y_pred)
        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
               tpr, thresholds

    def get_params(self):
        return self.best_params
