# Corporate Risk Analyzer

Arquitectura MLOps para la evaluación y clasificación de riesgos corporativos utilizando modelos basados en redes neuronales (FinBERT & Keras). 

Este proyecto utiliza **DVC** para el control de versiones de datos y **MLflow** (con backend en SQLite) para la trazabilidad y el registro de modelos.

## Quick Start

### 1. Clonar el repositorio y preparar el entorno
Clona el código fuente y crea un entorno virtual aislado para evitar conflictos de dependencias.

```bash
git clone https://github.com/cesar-upv/taller-dvc-mlflow
cd taller-dvc-mlflow
python -m venv .venv
source .venv/bin/activate  # En Windows usa: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Descargar los datos (DVC)

Los modelos preentrenados y los documentos pesados no se almacenan en Git. Utiliza DVC para descargarlos desde el almacenamiento remoto en la nube.

```bash
dvc pull
```

### 3. Ejecutar la Pipeline

Reproduce el flujo de trabajo completo (extracción, embeddings, predicción y evaluación). DVC leerá el archivo `dvc.lock` y ejecutará únicamente las etapas necesarias.

```bash
dvc repro
```

*Este comando generará automáticamente los reportes locales (`data/results.csv`) y la base de datos de MLflow (`mlflow.db`).*

### 4. Visualizar Resultados (MLflow)

Levanta el servidor frontend de MLflow apuntando a la base de datos local para explorar los experimentos, métricas de riesgo y el Model Registry.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre tu navegador en `http://127.0.0.1:5000` y selecciona el experimento **risk-analyzer-pipeline**.
