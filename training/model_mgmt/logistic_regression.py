from sklearn.linear_model import LogisticRegression

from utils.hyper_parameters_tuning import HyperParametersTuning
from utils.evaluation_metrics import EvaluationMetrics


class LogisticReg:

    def __init__(self):
        self.logistic_regression = LogisticRegression()
        self.hyper_param_tuning = HyperParametersTuning()
        self.evaluation_metric = EvaluationMetrics()

    def fit(self, x_train, y_train, params):
        self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                           y_train=y_train,
                                                                                           estimator=self.logistic_regression,
                                                                                           params=params)
        self.logistic_regression = self.best_estimator

        self.logistic_regression.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.logistic_regression.predict(x_test)
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
        tpr, thresholds = self.evaluation_metric.evaluate_performance(
            self.logistic_regression, x, y_test, y_pred)

        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds

    def get_params(self):
        return self.best_params
