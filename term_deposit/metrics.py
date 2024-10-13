from sklearn.metrics import recall_score

# Define a custom recall function for class 1 with logging
def recall_class_1_function(y_true, y_pred, **kwargs):
    # Print or log y_pred to check if class 1 is being predicted
    print("y_pred distribution:", pd.Series(y_pred).value_counts())  # Logging prediction distribution
    return recall_score(y_true, y_pred, pos_label=1, **kwargs)


# Define a custom recall function for class 1
class CustomRecallFunc:
    def map(self, pred, act, w, o, model):
        # Calculate recall for class 1
        true_positives = sum((act == 1) & (pred == 1))
        false_negatives = sum((act == 1) & (pred == 0))
        return [true_positives, true_positives + false_negatives]

    def reduce(self, l, r):
        return [l[0] + r[0], l[1] + r[1]]

    def metric(self, l):
        return l[0] / l[1] if l[1] > 0 else 0  # Recall = TP / (TP + FN)