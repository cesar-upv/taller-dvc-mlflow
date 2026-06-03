# Corporate Risk Analyzer

Arquitectura MLOps para la evaluación y clasificación de riesgos corporativos utilizando modelos basados en redes neuronales (FinBERT & Keras).

Este proyecto utiliza **DVC** para el control de versiones de datos y **MLflow Tracking Server** para la trazabilidad, métricas, artefactos y registro de experimentos.

MLflow puede ejecutarse localmente o en un servidor remoto mediante SSH tunneling. El cliente del taller se conecta al tracking server usando la variable de entorno `MLFLOW_TRACKING_URI`.

---

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

### 2. Descargar los datos con DVC

Los modelos preentrenados y los documentos pesados no se almacenan en Git. Utiliza DVC para descargarlos desde el almacenamiento remoto en la nube.

```bash
dvc pull
```

Esto descargará los archivos versionados por DVC, como:

```text
data/pdf/
models/
```

### 3. Configurar MLflow Tracking Server

Antes de ejecutar la pipeline, asegúrate de que el MLflow Tracking Server esté levantado.

#### Opción A: tracking server local

Si el tracking server está corriendo en tu misma máquina, configura el cliente de MLflow así:

```bash
export MLFLOW_TRACKING_URI=http://localhost:8080
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD='<password_generado_por_bootstrap>'
```

#### Opción B: tracking server remoto mediante SSH tunneling

Si el tracking server está corriendo en un servidor remoto, abre primero un túnel SSH desde tu máquina local:

```bash
ssh -L 8080:127.0.0.1:8080 user@server
```

Mientras ese túnel esté abierto, configura el cliente local de MLflow con los mismos exports:

```bash
export MLFLOW_TRACKING_URI=http://localhost:8080
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD='<password_generado_por_bootstrap>'
```

> Nota: `localhost:8080` apunta a tu propia máquina. Cuando usas SSH tunneling, ese puerto local se redirige al puerto `8080` del servidor remoto.

### 4. Ejecutar la pipeline

Reproduce el flujo de trabajo completo: extracción de texto, generación de embeddings, predicción, evaluación y registro de resultados en MLflow.

```bash
dvc repro
```

DVC leerá el archivo `dvc.lock` y ejecutará únicamente las etapas necesarias.

Este comando generará el reporte local:

```text
data/results.csv
```

y registrará en MLflow:

- parámetros del experimento
- métricas globales
- métricas por empresa
- artefactos generados
- gráficas de resumen de riesgo

### 5. Visualizar resultados en MLflow

Abre MLflow en:

```text
http://localhost:8080
```

Selecciona el experimento:

```text
risk-analyzer-pipeline
```

Los artefactos se guardan en el artifact store configurado en el tracking server, por ejemplo MinIO. El cliente del taller no necesita configurar credenciales de MinIO directamente porque el tracking server recibe y persiste los artefactos.

---

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

Esto actualizará el archivo `dvc.lock`, regenerará `data/results.csv` y registrará los nuevos resultados en el MLflow Tracking Server configurado mediante `MLFLOW_TRACKING_URI`.

### 3. Guardar la nueva versión: código y datos

Una vez que tu experimento termine, debes guardar ambos historiales: los datos pesados con DVC y el código/configuración con Git.

```bash
# 1. Guardar el registro de los datos pesados en DVC
dvc push

# 2. Guardar el código y el dvc.lock en Git
git add dvc.lock data/pdf.dvc
git commit -m "test: evaluacion de nuevos reportes financieros 10-K"
```

Si también cambiaste la configuración de MLflow o este README:

```bash
git add src/prediction.py README.md
git commit -m "feat: use MLflow tracking server"
```

### 4. Viajar entre versiones con `dvc checkout`

Si necesitas regresar a la rama `main` o a cualquier experimento anterior, cambiar la rama en Git **no es suficiente**, ya que Git no mueve los PDFs ni los modelos pesados.

Siempre que cambies de rama en Git, debes sincronizar los datos físicos con DVC:

```bash
# Paso 1: Moverte en el tiempo con Git: cambia scripts y dvc.lock
git checkout main

# Paso 2: Sincronizar los datos físicos: DVC lee el lock y descarga/acomoda los archivos pesados correctos
dvc checkout
```

---

## Variables de entorno relevantes

| Variable | Uso |
|---|---|
| `MLFLOW_TRACKING_URI` | URL del MLflow Tracking Server. Para local o SSH tunneling usa `http://localhost:8080`. |
| `MLFLOW_TRACKING_USERNAME` | Usuario de Basic Auth configurado en Nginx. |
| `MLFLOW_TRACKING_PASSWORD` | Password de Basic Auth generado por el script de bootstrap del tracking server. |

Ejemplo completo:

```bash
export MLFLOW_TRACKING_URI=http://localhost:8080
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD='<password_generado_por_bootstrap>'
dvc repro
```

---

## Notas sobre artifacts

Este taller no escribe directamente a MinIO ni necesita credenciales S3 locales para registrar artefactos.

El flujo esperado es:

```text
taller DVC/MLflow -> MLflow Tracking Server -> artifact store, por ejemplo MinIO
```

Esto funciona porque el tracking server está configurado para servir artefactos. Por eso, desde este repositorio solo necesitas configurar `MLFLOW_TRACKING_URI` y las credenciales de Basic Auth.
