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
    predicted_probs, risk_labels, title="Predicted risk probabilities"
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(risk_labels, predicted_probs, color="#3498db")
    ax.set_xlabel("Predicted probability")
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
        df["bert_embeddings"] = df["bert_embeddings"].apply(parse_embedding_string)
        X_test = np.vstack(df["bert_embeddings"].values)

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
                "sentence": risk_sentences["sentence"],
            }
        )

        for i, label in enumerate(risk_labels):
            df_risk[label] = predicted_probs[:, i]

        output_csv = "results.csv"
        df_risk.to_csv(output_csv, index=False)

        # Global metrics calculation
        global_summary = np.mean(predicted_probs, axis=0)

        # Logging global parameters to Parent Run
        mlflow.log_param("total_analyzed_sentences", len(df))
        mlflow.log_param("total_risk_sentences", len(idx))
        mlflow.log_param("total_companies_analyzed", len(df_risk["company"].unique()))
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

        print("Processing nested runs per company...")

        # 2. Start Child Runs (Per Company)
        companies = df_risk["company"].unique()

        for company in companies:
            run_name_child = f"evaluation_{company.lower()}"

            # nested=True connects this run to the parent_run
            with mlflow.start_run(run_name=run_name_child, nested=True):
                company_data = df_risk[df_risk["company"] == company]
                company_summary = company_data[risk_labels].mean()

                mlflow.log_param("company_ticker", company)
                mlflow.log_param("company_risk_sentences", len(company_data))

                for label, prob in zip(risk_labels, company_summary):
                    metric_name = label.replace(" ", "_").lower()
                    mlflow.log_metric(f"mean_{metric_name}", prob)

                # Generate and log an individual chart for the company
                fig_company = plot_risk_probabilities(
                    company_summary, risk_labels, f"Risk summary - {company.lower()}"
                )
                mlflow.log_figure(
                    fig_company, f"charts/risk_summary_{company.lower()}.png"
                )

        print("Evaluation completed. Global and nested runs saved to mlflow.")
