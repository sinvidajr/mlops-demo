from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()

model.fit(X, y)

with mlflow.start_run():

    mlflow.log_param(
        "modelo",
        "RandomForest"
    )

    mlflow.sklearn.log_model(
        model,
        "modelo"
    )

print("Modelo registrado")