from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Forzar una carpeta local compatible con GitHub Actions
mlflow.set_tracking_uri("file:./mlruns")

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