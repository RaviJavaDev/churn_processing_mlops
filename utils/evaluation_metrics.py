from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, \
    recall_score, roc_curve


class EvaluationMetrics:
    def __init__(self):
        pass

    def evaluate_performance(self, model, x, y_test, y_pred):
        score = model.score(x, y_test)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mtx = confusion_matrix(y_test, y_pred)
        roc_auc_scr = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_scores = f1_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        return score, classification_rep, confusion_mtx, roc_auc_scr, precision, recall, f1_scores, fpr, tpr, thresholds
