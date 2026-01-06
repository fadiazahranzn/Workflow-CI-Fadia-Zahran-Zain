import mlflow
import mlflow.sklearn

mlflow.set_experiment("CI_Retraining_Model")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
