from models.decision_trees import DecisionTrees
from models.logistic_regression import LogisticReg
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import matplotlib.pyplot as plt

from models.random_forest import RandomForest


class ModelSelection:
    def __init__(self, config):
        self.config = config

    def fit(self, x_train, y_train, x_test, y_test):

        estimators = self.config['estimators']
        mlflow_config = self.config['mlflow_config']
        mlflow.set_tracking_uri(mlflow_config['remote_server_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])

        for estimator in estimators:
            with mlflow.start_run(run_name=estimator, nested=True):
                if estimator == 'LogisticRegression':
                    params = estimators[estimator]['params']

                    model = LogisticReg()
                    model.fit(x_train=x_train, y_train=y_train, params=params)
                    y_pred = model.predict(x_test=x_test)
                    score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds = model.evaluate_performance(
                        x_test,
                        y_test,
                        y_pred)
                elif estimator == 'RandomForest':
                    params = estimators[estimator]['params']

                    model = RandomForest()
                    model.fit(x_train=x_train, y_train=y_train, params=params)
                    y_pred = model.predict(x_test=x_test)
                    score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds = model.evaluate_performance(
                        x_test,
                        y_test,
                        y_pred)
                elif estimator == 'DecisionTrees':
                    params = estimators[estimator]['params']

                    model = DecisionTrees()
                    model.fit(x_train=x_train, y_train=y_train, params=params)
                    y_pred = model.predict(x_test=x_test)
                    score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds = model.evaluate_performance(
                        x_test,
                        y_test,
                        y_pred)

                mlflow.log_params(model.get_params())
                self.log_metrics(mlflow, score, confusion_mtx, roc_auc, precision, recall, f1_score)
                self.log_artifact(mlflow, fpr, tpr, roc_auc)

                tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model,
                        estimator,
                        registered_model_name=mlflow_config['registered_model_name'])
                else:
                    mlflow.sklearn.load_model(model, estimator)

    def log_metrics(self, mlflow, score, confusion_mtx, roc_auc, precision, recall, f1_score):
        mlflow.log_metric('accuracy', score)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1_score)
        true_positive = confusion_mtx[0][0]
        false_positive = confusion_mtx[0][1]
        false_negative = confusion_mtx[1][0]
        true_negative = confusion_mtx[1][1]
        mlflow.log_metric('true_positive', true_positive)
        mlflow.log_metric('false_positive', false_positive)
        mlflow.log_metric('false_negative', false_negative)
        mlflow.log_metric('true_negative', true_negative)
        mlflow.log_metric('roc_auc_score', roc_auc)

    def log_artifact(self, mlflow, fpr, tpr, roc_auc):
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        # plt.show()
        plt.savefig("ROC_AUC_curve.png")
        mlflow.log_artifact("ROC_AUC_curve.png")
        plt.close()
