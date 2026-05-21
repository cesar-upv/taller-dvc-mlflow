import os
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

plt.style.use("default")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def parse_embedding_string(embedding_str):
    if isinstance(embedding_str, list):
        return embedding_str
    embedding_list_str = embedding_str.strip("[]").split()
    return [float(x) for x in embedding_list_str]


def plot_risk_probabilities(
    predicted_probs, risk_labels, title="Predicted Risk Probabilities"
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(risk_labels, predicted_probs, color="#3498db")
    ax.set_xlabel("Predicted Probability")
    ax.set_title(title)
    plt.tight_layout()
    return fig


###################################################################################################################
# DVC + MLflow Execution
###################################################################################################################
if __name__ == "__main__":
    # Strict MLflow configuration to save in the root directory
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlflow.set_tracking_uri(f"file://{os.path.join(ROOT_DIR, 'mlruns')}")
    mlflow.set_experiment("risk-analyzer-pipeline")

    print("Starting batch evaluation...")

    with mlflow.start_run(run_name="batch_evaluation"):
        # Load data
        df = pd.read_csv("data/embeddings.csv")
        df["bert_embeddings"] = df["bert_embeddings"].apply(parse_embedding_string)
        X_test = np.vstack(df["bert_embeddings"].values)

        # MODEL 1: Risk detection
        model1_path = "models/risk-detector.keras"
        model1 = tf.keras.models.load_model(model1_path)

        y_pred_probs = model1.predict(X_test)
        idx = np.where(y_pred_probs > 0.5)[0]
        print(f"Oraciones de riesgo detectadas: {len(idx)} de {len(df)}")

        # MODEL 2: Risk classification
        risk_labels = [
            "Political Risk",
            "Financial Risk",
            "Interest Rate Risk",
            "Country Risk",
            "Social Risk",
            "Environmental Risk",
            "Operational Risk",
            "Management Risk",
            "Legal Risk",
            "Competition",
            "Reputational Risk",
        ]
        model2_path = "models/risk-classifier.keras"
        model2 = tf.keras.models.load_model(model2_path)

        X_risk_only = X_test[idx]
        risk_sentences = df.iloc[idx].reset_index(drop=True)
        predicted_probs = model2.predict(X_risk_only)

        # Create final DataFrame for risk sentences
        df_risk = pd.DataFrame(
            {
                "company": risk_sentences["company"],
                "sentence": risk_sentences["sentence"],
            }
        )

        for i, label in enumerate(risk_labels):
            df_risk[label] = predicted_probs[:, i]

        output_csv = "results.csv"
        df_risk.to_csv(output_csv, index=False)

        global_summary = np.mean(predicted_probs, axis=0)

        # Obtain the current Git commit hash
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except Exception:
            git_commit = "Desconocido"

        mlflow.set_tag("commit_hash_dvc", git_commit)

        # Logging to MLflow
        mlflow.log_param("total_analyzed_sentences", len(df))
        mlflow.log_param("total_risk_sentences", len(idx))
        mlflow.log_param("filter_model", model1_path)
        mlflow.log_param("classifier_model", model2_path)

        for label, prob in zip(risk_labels, global_summary):
            metric_name = label.replace(" ", "_").lower()
            mlflow.log_metric(f"mean_{metric_name}", prob)

        # Artifacts
        mlflow.log_artifact(output_csv, artifact_path="generated_reports")

        fig = plot_risk_probabilities(
            global_summary, risk_labels, "Global Risk Summary"
        )
        mlflow.log_figure(fig, "charts/global_risk_summary.png")

        print("Evaluation completed. Results saved to MLflow.")
