from utils.evaluation_metrics import EvaluationMetrics
from utils.hyper_parameters_tuning import HyperParametersTuning
from sklearn.tree import DecisionTreeClassifier


class DecisionTrees:
    def __init__(self, logger):
        self.logger = logger
        self.decision_trees = DecisionTreeClassifier()
        self.hyper_param_tuning = HyperParametersTuning(self.logger)
        self.evaluation_metric = EvaluationMetrics(self.logger)
        self.best_estimator = None
        self.best_params = None

    def fit(self, x_train, y_train, params):
        self.logger.info('***** DecisionTrees Model building Started *****')
        try:
            self.best_estimator, self.best_params = self.hyper_param_tuning.get_best_estimator(x_train=x_train,
                                                                                               y_train=y_train,
                                                                                               estimator=self.decision_trees,
                                                                                               params=params)
            self.decision_trees = self.best_estimator

            self.decision_trees.fit(x_train, y_train)
        except Exception as e:
            self.logger.error(f'error in DecisionTrees fit e: {e}')
            raise e
        self.logger.info('***** DecisionTrees Model building Finished *****')

    def predict(self, x_test):
        self.logger.info('***** In DecisionTrees Predict Started *****')
        try:
            y_pred = self.decision_trees.predict(x_test)
        except Exception as e:
            self.logger.error(f'error in DecisionTrees predict e: {e}')
            raise e
        self.logger.info('***** In DecisionTrees Predict Finished *****')
        return y_pred

    def evaluate_performance(self, x, y_test, y_pred):
        self.logger.info('***** In DecisionTrees evaluate_performance Started *****')
        try:
            score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
            tpr, thresholds = self.evaluation_metric.evaluate_performance(
                self.decision_trees, x, y_test, y_pred)
        except Exception as e:
            self.logger.error(f'error in DecisionTrees evaluate_performance e: {e}')
            raise e
        self.logger.info('***** In DecisionTrees evaluate_performance Finished *****')
        return score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, \
               tpr, thresholds

    def get_params(self):
        self.logger.info('***** In DecisionTrees get_params *****')
        return self.best_params
