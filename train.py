import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---

from pathlib import Path

workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")

tracking_uri = Path(mlruns_dir).resolve().as_uri()
artifact_location = Path(mlruns_dir).resolve().as_uri()

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# --- Asegurar que el directorio MLRuns exista ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Manejo correcto del experimento ---
experiment_name = "CI-CD-Lab2"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"--- Debug: Creando experimento '{experiment_name}' ---")
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
else:
    print(f"--- Debug: Experimento ya existe, usando el existente ---")
    experiment_id = experiment.experiment_id

print(f"--- Debug: Usando experimento ID: {experiment_id} ---")

# Asegurarse de que tenemos un experiment_id válido
if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}'. ---")
    sys.exit(1)

# --- Cargar Datos y Entrenar Modelo ---
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
import joblib
joblib.dump(model, "model.pkl")
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Iniciar Run de MLflow ---
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---") # Añadir ID aquí
run = None
try:
    # Iniciar el run PASANDO EXPLÍCITAMENTE el experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run: # <--- CAMBIO CLAVE
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        # Comprobar si coincide con el patrón esperado basado en artifact_location del experimento
        # (La artifact_uri del run incluirá el run_id)
        expected_artifact_uri_base = os.path.join(artifact_location, run_id, "artifacts")
        if actual_artifact_uri != expected_artifact_uri_base:
             print(f"--- WARNING: La URI del Artefacto del Run '{actual_artifact_uri}' no coincide exactamente con la esperada '{expected_artifact_uri_base}' (esto puede ser normal si la estructura difiere ligeramente). Lo importante es que NO sea la ruta local incorrecta. ---")
        if "/home/manuelcastiblan/" in actual_artifact_uri:
             print(f"--- ¡¡¡ERROR CRÍTICO!!!: La URI del Artefacto del Run '{actual_artifact_uri}' TODAVÍA contiene la ruta local incorrecta! ---")


        mlflow.log_metric("mse", mse)
        print(f"--- Debug: Intentando log_model con artifact_path='model' ---")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        print(f"✅ Modelo registrado correctamente. MSE: {mse:.4f}")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID intentado: {experiment_id}") # Añadir ID aquí
    if run:
         print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}")
    else:
         print("El objeto Run no se creó con éxito.")
    sys.exit(1)
