# Corporate Risk Analyzer

Arquitectura MLOps para la evaluación y clasificación de riesgos corporativos utilizando modelos basados en redes neuronales (FinBERT & Keras). 

Este proyecto utiliza **DVC** para el control de versiones de datos y **MLflow** (con backend en SQLite) para la trazabilidad y el registro de modelos.

## Quick start

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

### 3. Ejecutar la pipeline
Reproduce el flujo de trabajo completo (extracción, embeddings, predicción y evaluación). DVC leerá el archivo `dvc.lock` y ejecutará únicamente las etapas necesarias.

```bash
dvc repro
```

*Este comando generará automáticamente los reportes locales (`data/results.csv`) y la base de datos de MLflow (`mlflow.db`).*

### 4. Visualizar resultados (MLflow)
Levanta el servidor frontend de MLflow apuntando a la base de datos local para explorar los experimentos, métricas de riesgo y el Model Registry.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre tu navegador en `http://127.0.0.1:5000` y selecciona el experimento **risk-analyzer-pipeline**.

## Experimentación y versionado

### 1. Crear una rama aislada para pruebas
Si eres un tester o quieres evaluar nuevos documentos financieros, primero crea tu propio espacio de trabajo en Git:

```bash
git checkout -b test/nuevos-reportes
```

### 2. Procesar nuevos datos

1. Agrega tus nuevos archivos PDF a la carpeta `data/pdf/`.
2. Ejecuta el pipeline para extraer el texto, generar los embeddings y evaluar el modelo con los nuevos datos:

```bash
dvc repro
```

*Esto actualizará el archivo `dvc.lock` y registrará los nuevos resultados en SQLite.*

### 3. Guardar la nueva versión (Código + Datos)
Una vez que tu experimento termine, debes guardar ambos historiales:

```bash
# 1. Guardar el registro de los datos pesados en DVC
dvc push

# 2. Guardar el código y el dvc.lock en Git
git add dvc.lock data/pdf.dvc
git commit -m "test: evaluacion de nuevos reportes financieros 10-K"
```

### 4. Viajar entre versiones (`dvc checkout`)
Si necesitas regresar a la rama `main` (o a cualquier experimento anterior), cambiar la rama en Git **no es suficiente**, ya que Git no mueve los PDFs ni los modelos pesados.

Siempre que cambies de rama en Git, debes sincronizar los datos con DVC:

```bash
# Paso 1: Moverte en el tiempo con Git (cambia los scripts y el dvc.lock)
git checkout main

# Paso 2: Sincronizar los datos físicos (DVC lee el lock y descarga/acomoda los archivos pesados correctos)
dvc checkout
```
