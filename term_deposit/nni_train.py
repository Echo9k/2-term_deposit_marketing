import nni
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Get parameters from NNI
params = nni.get_next_parameter()
C = params.get("C", 1.0)
max_iter = params.get("max_iter", 100)

# Initialize MLflow
mlflow.set_experiment("customer_subscription_classification_nni")

with mlflow.start_run():
    # Train model
    logreg = LogisticRegression(random_state=42, class_weight='balanced', C=C, max_iter=max_iter)
    logreg.fit(X_train_sm, y_train_sm)

    # Predict and evaluate
    y_pred_test = logreg.predict(X_test)
    roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

    # Log parameters and metrics to MLflow
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("roc_auc", roc_auc)

    # Report results to NNI
    nni.report_final_result(roc_auc)

    print(f"ROC AUC Score: {roc_auc}")
    print(classification_report(y_test, y_pred_test))
