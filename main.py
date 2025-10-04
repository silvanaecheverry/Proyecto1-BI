from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.metrics import classification_report
import pandas as pd

# Inicializar la app
app = FastAPI()

# Cargar el modelo guardado
modelo = joblib.load("Proyecto1-BI/modelo.pkl")


# Definir esquema de entrada para /predict
class PredictRequest(BaseModel):
    textos: list[str]

# Definir esquema de entrada para /retrain
class RetrainRequest(BaseModel):
    textos: list[str]
    labels: list[int]

# Endpoint básico
@app.get("/")
def home():
    return {"mensaje": "API Proyecto1 funcionando "}

# Endpoint 1: predicción
@app.post("/predict")
def predict(req: PredictRequest):
    predicciones = modelo.predict(req.textos)
    return {"predicciones": predicciones.tolist()}

# Endpoint 2: reentrenamiento
@app.post("/retrain")
def retrain(req: RetrainRequest):
    global modelo

    nuevos_X = req.textos
    nuevos_y = req.labels

    if len(nuevos_X) != len(nuevos_y):
        return {"error": "El número de textos y labels no coincide"}

    # Leer dataset base
    data_original = pd.read_csv("dataset_base.csv")

    # Agregar nuevos datos
    nuevos_df = pd.DataFrame({"textos": nuevos_X, "labels": nuevos_y})
    data_total = pd.concat([data_original, nuevos_df], ignore_index=True)

    # Guardar dataset actualizado
    data_total.to_csv("dataset_base.csv", index=False)

    # Reentrenar modelo con todos los datos acumulados
    modelo.fit(data_total["textos"], data_total["labels"])

    joblib.dump(modelo, "modelo.pkl")

    # Métricas rápidas
    y_pred = modelo.predict(data_total["textos"])
    reporte = classification_report(data_total["labels"], y_pred, output_dict=True)

    return {
        "mensaje": "Modelo reentrenado y dataset actualizado ",
        "precision": reporte["weighted avg"]["precision"],
        "recall": reporte["weighted avg"]["recall"],
        "f1_score": reporte["weighted avg"]["f1-score"]
    }
