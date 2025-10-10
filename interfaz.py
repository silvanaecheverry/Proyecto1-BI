# ============================================
# INTERFAZ Y API - PROYECTO 1 ETAPA 2 - BI
# ============================================
import os
import pandas as pd
import joblib
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ============================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================

# Construir ruta absoluta
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "dataset_base.csv")
modelo_path = os.path.join(base_dir, "Proyecto1-BI", "modelo.pkl")

# Crear carpetas necesarias
os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "Proyecto1-BI"), exist_ok=True)

# ============================================
# 2. CARGA DEL MODELO Y DEL DATASET
# ============================================

# Cargar o crear modelo
if os.path.exists(modelo_path):
    modelo = joblib.load(modelo_path)
    print("Modelo cargado correctamente desde:", modelo_path)
else:
    print("No se encontró modelo.pkl, creando uno nuevo...")
    modelo = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=1, max_df=1.0)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    df_temp = pd.DataFrame({"textos": ["texto ejemplo"], "labels": [0]})
    modelo.fit(df_temp["textos"], df_temp["labels"])
    joblib.dump(modelo, modelo_path)
    print("Modelo base creado y guardado en:", modelo_path)

# Crear dataset vacío si no existe
if not os.path.exists(data_path):
    pd.DataFrame(columns=["textos", "labels"]).to_csv(data_path, index=False)
    print("dataset_base.csv creado vacío en carpeta data/.")
else:
    print(f"Intentando cargar dataset desde: {data_path}")

# Verificar dataset actual
try:
    df_check = pd.read_csv(
        data_path,
        sep=",",
        engine="python",
        on_bad_lines="skip",
        quoting=3,
        encoding="utf-8"
    )

    n_filas = len(df_check)
    n_clases = df_check["labels"].nunique()

    print(f"Dataset cargado correctamente ({n_filas} filas, {n_clases} clases únicas).")
    print(f"Columnas detectadas: {list(df_check.columns)}")

    if n_filas < 100:
        print("Advertencia: el dataset parece pequeño (<100 filas). Verifica si contiene todos los datos.")

except Exception as e:
    print(f"Error al leer dataset_base.csv: {e}")

# ============================================
# 3. DEFINICIÓN DE LA API REST (FastAPI)
# ============================================

app = FastAPI(title="API Proyecto 1 - Etapa 2")

class PredictRequest(BaseModel):
    textos: list[str]

class RetrainRequest(BaseModel):
    textos: list[str]
    labels: list[int]

# --- Endpoint 1: Predicción ---
@app.post("/predict")
def predict(req: PredictRequest):
    global modelo
    textos = req.textos
    preds = modelo.predict(textos).tolist()
    return {"predicciones": preds}

# --- Endpoint 2: Reentrenamiento ---
@app.post("/retrain")
def retrain(req: RetrainRequest):
    global modelo
    nuevos_X = req.textos
    nuevos_y = req.labels

    if len(nuevos_X) != len(nuevos_y):
        return {"error": "El número de textos y etiquetas debe coincidir."}

    df = pd.read_csv(data_path, sep=",", engine="python", on_bad_lines="skip", quoting=3)
    df_nuevos = pd.DataFrame({"textos": nuevos_X, "labels": nuevos_y})
    df = pd.concat([df, df_nuevos], ignore_index=True)
    df.to_csv(data_path, index=False)

    if df["labels"].nunique() < 2:
        return {"error": "El modelo necesita al menos 2 clases diferentes para entrenar."}

    modelo = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=1, max_df=1.0)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    modelo.fit(df["textos"], df["labels"])
    joblib.dump(modelo, modelo_path)
    modelo = joblib.load(modelo_path)  # Recarga el nuevo modelo entrenado
    y_pred = modelo.predict(df["textos"])
    report = classification_report(df["labels"], y_pred, output_dict=True, zero_division=0)

    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "total_ejemplos": len(df)
    }

# ============================================
# 4. INTERFAZ DE USUARIO CON GRADIO
# ============================================

def predecir_texto(texto):
    if not texto.strip():
        return "Por favor ingrese un texto válido."
    try:
        pred = modelo.predict([texto])[0]
        prob = modelo.predict_proba([texto]).max()
    except Exception:
        return "El modelo aún no tiene suficientes datos para predecir."
    return (
        f" **Texto ingresado:** {texto}\n\n"
        f" **Predicción del modelo:** {pred}\n"
        f" **Nivel de confianza:** {prob*100:.2f}%"
    )

def reentrenar_modelo(textos, etiquetas):
    textos_list = [t.strip() for t in textos.split('\n') if t.strip()]
    etiquetas_list = [e.strip() for e in etiquetas.replace(',', ' ').split() if e.strip()]

    if len(textos_list) != len(etiquetas_list):
        return f"Debes ingresar la misma cantidad de textos ({len(textos_list)}) y etiquetas ({len(etiquetas_list)})."

    try:
        etiquetas_list = [int(e) for e in etiquetas_list]
    except ValueError:
        return "Todas las etiquetas deben ser números enteros (0, 1, 2, 3, ...)."

    df = pd.read_csv(data_path, sep=",", engine="python", on_bad_lines="skip", quoting=3)
    print(f"Dataset actual: {len(df)} registros antes de agregar nuevos.")

    nuevos = pd.DataFrame({"textos": textos_list, "labels": etiquetas_list})
    df = pd.concat([df, nuevos], ignore_index=True)
    df.to_csv(data_path, index=False)

    if df["labels"].nunique() < 2:
        return "Se necesitan al menos 2 clases para entrenar."

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=1, max_df=1.0)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(df["textos"], df["labels"])
    joblib.dump(pipeline, modelo_path)

    pred = pipeline.predict(df["textos"])
    rep = classification_report(df["labels"], pred, output_dict=True, zero_division=0)
    p, r, f1 = rep["weighted avg"]["precision"], rep["weighted avg"]["recall"], rep["weighted avg"]["f1-score"]

    return (
        f"**Modelo reentrenado con {len(nuevos)} nuevos ejemplos.**\n\n"
        f"**Métricas globales (total {len(df)} registros):**\n"
        f"- Precisión: {p:.2f}\n- Recall: {r:.2f}\n- F1-Score: {f1:.2f}"
    )

interfaz_prediccion = gr.Interface(
    fn=predecir_texto,
    inputs=gr.Textbox(label=" Escribe un texto para analizar"),
    outputs=gr.Markdown(label=" Resultado del modelo"),
    title=" Predicción de Textos",
    description="Ingresa un texto y obtén la categoría predicha con el modelo actual.\n\n"
        " **Categorías del modelo:**\n"
        "- 1 → Fin de la pobreza (ODS 1)\n"
        "- 3 → Salud y bienestar (ODS 3)\n"
        "- 4 → Educación de calidad (ODS 4)"
)

interfaz_reentrenamiento = gr.Interface(
    fn=reentrenar_modelo,
    inputs=[
        gr.Textbox(lines=6, label=" Nuevos textos (uno por línea)"),
        gr.Textbox(label=" Etiquetas correspondientes (separadas por coma o espacio)")
    ],
    outputs=gr.Markdown(label=" Resultado del reentrenamiento"),
    title=" Reentrenar modelo (Usuario experto)",
    description="Agrega varios textos con sus etiquetas para actualizar el modelo y mejorar su aprendizaje.\n\n"
        " **Categorías del modelo:**\n"
        "- 1 → Fin de la pobreza (ODS 1)\n"
        "- 3 → Salud y bienestar (ODS 3)\n"
        "- 4 → Educación de calidad (ODS 4)"
)

demo = gr.TabbedInterface(
    [interfaz_prediccion, interfaz_reentrenamiento],
    ["Predicción", "Reentrenar modelo"]
)
print("\n Categorías del modelo:")
print("1 → Fin de la pobreza (ODS 1)")
print("3 → Salud y bienestar (ODS 3)")
print("4 → Educación de calidad (ODS 4)\n")


# ============================================
# 5. EJECUCIÓN LOCAL
# ============================================
if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, server_port=7860)
