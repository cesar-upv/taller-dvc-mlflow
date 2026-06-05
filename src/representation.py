import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import glob

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from preprocessing import clean_text, split_into_sentences


def get_financial_bert_fine_tuned_embeddings(batch_sentences, tokenizerr, modell):
    if isinstance(batch_sentences, str):
        batch_sentences = [batch_sentences]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modell = modell.to(device)

    encoded = tokenizerr(
        batch_sentences,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: tensor.to(device) for key, tensor in encoded.items()}

    with torch.no_grad():
        last_hidden_states = modell(**encoded)

    bert_embeddings = last_hidden_states.last_hidden_state[:, 0, :].cpu().numpy()

    return bert_embeddings


def process_document_in_batches(sentences, tokenizer, model, batch_size=32):
    sentence_batches = [
        sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)
    ]
    num_batches = len(sentence_batches)
    sentence_embeddings = [None] * len(sentences)

    for i, batch in enumerate(sentence_batches, start=1):
        batch_embeddings = get_financial_bert_fine_tuned_embeddings(
            batch, tokenizer, model
        )
        offset = (i - 1) * batch_size
        for j, embedding in enumerate(batch_embeddings):
            sentence_embeddings[offset + j] = embedding
        print(f"   Processing batch {i}/{num_batches}...")
    return sentence_embeddings


###################################################################################################################
# DVC Pipeline Execution
###################################################################################################################
if __name__ == "__main__":
    RISK_TXT_DIR = "data/risk-txt"
    OUTPUT_CSV = "data/embeddings.csv"
    MODEL_PATH = "models/financial-bert-mlm"

    print("Loading Financial BERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)

    all_data = []
    txt_files = glob.glob(os.path.join(RISK_TXT_DIR, "*.txt"))

    for txt_path in txt_files:
        # Extract company ticker from filename
        filename = os.path.basename(txt_path)
        company_ticker = filename.split("_")[0]
        document_id = os.path.splitext(filename)[0]

        print(f"Processing embeddings for: {document_id}")

        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        # Cleaning and splitting using preprocessing functions
        cleaned_text = clean_text(text)
        sentences = split_into_sentences(cleaned_text)

        if not sentences:
            continue

        embeddings = process_document_in_batches(sentences, tokenizer, model)

        for sentence, embedding in zip(sentences, embeddings):
            all_data.append(
                {
                    "company": company_ticker,  # <--- Aquí usamos el Ticker limpio
                    "document_id": document_id,
                    "sentence": sentence,
                    "bert_embeddings": embedding.tolist(),
                }
            )

    print("Saving unified CSV...")
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Embeddings generation finished.")
