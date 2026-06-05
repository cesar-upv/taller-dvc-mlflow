import ast
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

    # Convert to safe list
    try:
        return ast.literal_eval(embedding_str)
    except (ValueError, SyntaxError):
        # Fallback
        return []


def plot_risk_probabilities(
    predicted_probs, risk_labels, title="Predicted risk probabilities"
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(risk_labels, predicted_probs, color="#3498db")
    ax.set_xlabel("Predicted probability")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def slugify_mlflow_name(value):
    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


###################################################################################################################
# DVC + MLflow Execution
###################################################################################################################
if __name__ == "__main__":
    # MLflow Tracking Server configuration
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("risk-analyzer-pipeline")

    print("Starting batch evaluation...")

    # Obtain the current Git commit hash
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        git_commit = "UNKNOWN"

    # 1. Start Parent Run (Global Portfolio)
    with mlflow.start_run(run_name="batch_evaluation") as parent_run:
        mlflow.set_tag("commit_hash_dvc", git_commit)

        # Load data
        df = pd.read_csv("data/embeddings.csv")
        if "document_id" not in df.columns:
            df["document_id"] = df["company"]
        df["bert_embeddings"] = df["bert_embeddings"].apply(parse_embedding_string)
        X_test = np.vstack(df["bert_embeddings"].values)

        # Log the global dataset as an input artifact
        dataset_global = mlflow.data.from_pandas(
            df, source="data/embeddings.csv", name="full_embeddings"
        )
        mlflow.log_input(dataset_global, context="batch_inference")

        # MODEL 1: Risk detection
        model1_path = "models/risk-detector.keras"
        model1 = tf.keras.models.load_model(model1_path)

        y_pred_probs = model1.predict(X_test)
        idx = np.where(y_pred_probs > 0.5)[0]
        print(f"Risk sentences detected: {len(idx)} out of {len(df)}")

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
                "document_id": risk_sentences["document_id"],
                "sentence": risk_sentences["sentence"],
            }
        )

        for i, label in enumerate(risk_labels):
            df_risk[label] = predicted_probs[:, i]

        output_csv = "data/results.csv"
        df_risk.to_csv(output_csv, index=False)

        # Global metrics calculation
        global_summary = np.mean(predicted_probs, axis=0)

        # Logging global parameters to Parent Run
        mlflow.log_param("total_analyzed_sentences", len(df))
        mlflow.log_param("total_risk_sentences", len(idx))
        mlflow.log_param("total_companies_analyzed", len(df["company"].unique()))
        mlflow.log_param("total_documents_analyzed", len(df["document_id"].unique()))
        mlflow.log_param("total_companies_with_risk", len(df_risk["company"].unique()))
        mlflow.log_param(
            "total_documents_with_risk", len(df_risk["document_id"].unique())
        )
        mlflow.log_param("filter_model", model1_path)
        mlflow.log_param("classifier_model", model2_path)

        # Logging global metrics
        for label, prob in zip(risk_labels, global_summary):
            metric_name = label.replace(" ", "_").lower()
            mlflow.log_metric(f"global_mean_{metric_name}", prob)

        # Artifacts (global)
        mlflow.log_artifact(output_csv, artifact_path="generated_reports")
        fig = plot_risk_probabilities(
            global_summary, risk_labels, "Global risk summary"
        )
        mlflow.log_figure(fig, "charts/global_risk_summary.png")

        print("Processing nested runs per document...")

        # 2. Start Child Runs (Per Document)
        documents = df_risk[["company", "document_id"]].drop_duplicates()

        for _, document in documents.iterrows():
            company = document["company"]
            document_id = document["document_id"]
            document_slug = slugify_mlflow_name(document_id)
            run_name_child = f"evaluation_{document_slug}"

            # nested=True connects this run to the parent_run
            with mlflow.start_run(run_name=run_name_child, nested=True):
                document_data = df_risk[df_risk["document_id"] == document_id]

                # Log the document dataset as an input artifact
                dataset_document = mlflow.data.from_pandas(
                    document_data, name=f"dataset_{document_slug}"
                )
                mlflow.log_input(dataset_document, context="document_inference")

                document_summary = document_data[risk_labels].mean()

                mlflow.log_param("company_ticker", company)
                mlflow.log_param("document_id", document_id)
                mlflow.log_param("document_risk_sentences", len(document_data))

                for label, prob in zip(risk_labels, document_summary):
                    metric_name = label.replace(" ", "_").lower()
                    mlflow.log_metric(f"mean_{metric_name}", prob)

                # Generate and log an individual chart for the document
                fig_document = plot_risk_probabilities(
                    document_summary, risk_labels, f"Risk summary - {document_id}"
                )
                mlflow.log_figure(
                    fig_document, f"charts/risk_summary_{document_slug}.png"
                )

        print("Evaluation completed. Global and nested runs saved to mlflow.")
