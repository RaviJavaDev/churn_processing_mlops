from training.model_mgmt.decision_trees import DecisionTrees
from training.model_mgmt.light_gbm import LightGBM
from training.model_mgmt.logistic_regression import LogisticReg
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

from training.model_mgmt.random_forest import RandomForest

from lightgbm import plot_importance


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
                    self.log_roc_auc_artifact(model.logistic_regression, x_test=x_test, y_test=y_test)

                    self.log_confusion_matrix_artifact(model.logistic_regression, x_test=x_test, y_test=y_test)
                elif estimator == 'RandomForest':
                    params = estimators[estimator]['params']

                    model = RandomForest()
                    model.fit(x_train=x_train, y_train=y_train, params=params)
                    y_pred = model.predict(x_test=x_test)
                    score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds = model.evaluate_performance(
                        x_test,
                        y_test,
                        y_pred)
                    self.log_roc_auc_artifact(model.random_forest, x_test=x_test, y_test=y_test)

                    self.log_confusion_matrix_artifact(model.random_forest, x_test=x_test, y_test=y_test)
                elif estimator == 'DecisionTrees':
                    params = estimators[estimator]['params']

                    model = DecisionTrees()
                    model.fit(x_train=x_train, y_train=y_train, params=params)
                    y_pred = model.predict(x_test=x_test)
                    score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds = model.evaluate_performance(
                        x_test,
                        y_test,
                        y_pred)
                    self.log_roc_auc_artifact(model.decision_trees, x_test=x_test, y_test=y_test)
                    self.log_confusion_matrix_artifact(model.decision_trees, x_test=x_test, y_test=y_test)
                elif estimator == 'LightGBM':
                    params = estimators[estimator]['params']

                    model = LightGBM()
                    model.fit(x_train=x_train, y_train=y_train, params=params)
                    y_pred = model.predict(x_test=x_test)
                    score, classification_rep, confusion_mtx, roc_auc, precision, recall, f1_score, fpr, tpr, thresholds = model.evaluate_performance(
                        x_test,
                        y_test,
                        y_pred)
                    self.log_roc_auc_artifact(model.lbm_classifier, x_test=x_test, y_test=y_test)

                    self.log_confusion_matrix_artifact(model.lbm_classifier, x_test=x_test, y_test=y_test)

                    self.log_feature_importance_artifact(model.lbm_classifier)

                mlflow.log_params(model.get_params())

                self.log_metrics(mlflow, score, confusion_mtx, roc_auc, precision, recall, f1_score)

                mlflow.log_artifact('./requirements.txt')

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

    def log_roc_auc_artifact(self, model, x_test, y_test):

        # Plot and save AUC details
        plot_roc_curve(model, x_test, y_test)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC AUC Curve')
        filename = f'./images/validation_roc_curve.png'
        plt.savefig(filename)
        # log model artifacts
        mlflow.log_artifact(filename)

    def log_confusion_matrix_artifact(self, model, x_test, y_test):
        plot_confusion_matrix(model, x_test, y_test,
                              display_labels=['Placed', 'Not Placed'],
                              cmap='magma')
        plt.title('Confusion Matrix')
        filename = f'./images/validation_confusion_matrix.png'
        plt.savefig(filename)
        # log model artifacts
        mlflow.log_artifact(filename)

    def log_feature_importance_artifact(self, model):
        plot_importance(model, height=0.4)
        filename = './images/lgb_validation_feature_importance.png'
        plt.savefig(filename)
        # log model artifacts
        mlflow.log_artifact(filename)
