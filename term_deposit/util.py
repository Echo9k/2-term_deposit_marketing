from sklearn.metrics import recall_score

# Define a custom recall function for class 1 with logging
def recall_class_1_function(y_true, y_pred, **kwargs):
    # Print or log y_pred to check if class 1 is being predicted
    print("y_pred distribution:", pd.Series(y_pred).value_counts())  # Logging prediction distribution
    return recall_score(y_true, y_pred, pos_label=1, **kwargs)