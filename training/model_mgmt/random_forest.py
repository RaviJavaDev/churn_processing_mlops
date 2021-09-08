from sklearn.ensemble import RandomForestClassifier

from utils.evaluation_metrics import EvaluationMetrics
from utils.hyper_parameters_tuning import HyperParametersTuning


class RandomForest:
    def __init__(self):
        self.random_forest = RandomForestClassifier()
        self.hyper_param_tuning = HyperParametersTuning()
        self.evaluation_metric = EvaluationMetrics()

    def fit(self, x_train, y_train, params):
        self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                           y_train=y_train,
                                                                                           estimator=self.random_forest,
                                                                                           params=params)
        self.random_forest = self.best_estimator

        self.random_forest.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.random_forest.predict(x_test)
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
        tpr, thresholds = self.evaluation_metric.evaluate_performance(
            self.random_forest, x, y_test, y_pred)
        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
               tpr, thresholds

    def get_params(self):
        return self.best_params
