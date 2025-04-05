import os
import re
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ============================================================================
# 0. Configuración y Ajustes
# ============================================================================
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ============================================================================
# 1. Entrenamiento y Predicción 
# ============================================================================
def train_and_predict(seed, rt_labeled, rt_unlabeled):
    """
    Entrena el modelo con una semilla dada, predice en el dataset de Rotten Tomatoes
    y retorna las predicciones.
    """
    set_seed(seed)
    imdb = load_dataset("imdb")
    train_ds, test_ds = imdb["train"], imdb["test"]

    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_fn(batch):
        # Limpia cada texto y lo tokeniza
        cleaned = [clean_text(t) for t in batch["text"]]
        return tokenizer(cleaned, padding="max_length", truncation=True, max_length=512)

    # Preprocesamiento de IMDB
    train_ds = train_ds.map(tokenize_fn, batched=True).rename_column("label", "labels").remove_columns(["text"])
    test_ds = test_ds.map(tokenize_fn, batched=True).rename_column("label", "labels").remove_columns(["text"])
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    # =========================================================================
    # Ajustes del Entrenamiento
    # =========================================================================
    training_args = TrainingArguments(
        output_dir=f"./model_output_seed{seed}",
        num_train_epochs=10,
        learning_rate=1.5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,
        warmup_ratio=0.05,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Re-cargar el mejor checkpoint
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        model = AutoModelForSequenceClassification.from_pretrained(best_ckpt, num_labels=2)
        if torch.cuda.is_available():
            model = model.to("cuda")
        trainer.model = model

    # =========================================================================
    # Predicción con labels
    # =========================================================================
    ds_rt_labeled = Dataset.from_pandas(rt_labeled)
    ds_rt_labeled = ds_rt_labeled.map(tokenize_fn, batched=True).rename_column("pred", "labels")
    ds_rt_labeled = ds_rt_labeled.remove_columns([c for c in ["text", "id", "pred"] if c in ds_rt_labeled.column_names])
    ds_rt_labeled.set_format("torch")
    preds_rt = trainer.predict(ds_rt_labeled).predictions
    preds_rt_labels = np.argmax(preds_rt, axis=1)

    # =========================================================================
    # Predicción sin labels
    # =========================================================================
    ds_rt_unlabeled = Dataset.from_pandas(rt_unlabeled)
    ds_rt_unlabeled = ds_rt_unlabeled.map(tokenize_fn, batched=True)
    rt_ids = ds_rt_unlabeled["id"]
    ds_rt_unlabeled = ds_rt_unlabeled.remove_columns([c for c in ["text", "id"] if c in ds_rt_unlabeled.column_names])
    ds_rt_unlabeled.set_format("torch")
    preds_rt_unlab = trainer.predict(ds_rt_unlabeled).predictions
    preds_rt_unlab_labels = np.argmax(preds_rt_unlab, axis=1)

    return preds_rt_labels, preds_rt_unlab_labels, rt_ids

# ============================================================================
# 2. Ensamble y CSV 
# ============================================================================
def main():
    # Cargar datasets de Rotten Tomatoes
    rt_labeled = pd.read_csv(os.path.join("data", "rt_test_with_labels.csv"))
    rt_unlabeled = pd.read_csv(os.path.join("data", "rt_test_unlabeled.csv"))

    # Entrenar con distintas semillas
    seed_list = [42, 2023, 999]
    all_preds_labeled = []
    all_preds_unlabeled = []

    for seed in seed_list:
        print(f"\n===== Entrenando con semilla: {seed} =====")
        preds_rt_labels, preds_rt_unlab_labels, rt_ids = train_and_predict(seed, rt_labeled, rt_unlabeled)
        all_preds_labeled.append(preds_rt_labels)
        all_preds_unlabeled.append(preds_rt_unlab_labels)

    # Ensemble por votación
    ensemble_preds_labeled = np.array(all_preds_labeled).T  
    final_labeled = [np.bincount(row).argmax() for row in ensemble_preds_labeled]
    acc_ens = accuracy_score(rt_labeled["pred"], final_labeled)
    print("\n===== ENSEMBLE =====")
    print(f"Accuracy en Rotten Tomatoes (con labels) [Ensemble]: {acc_ens:.4f}")

    # Ensemble para Rotten Tomatoes 
    ensemble_preds_unlabeled = np.array(all_preds_unlabeled).T
    final_unlabeled = [np.bincount(row).argmax() for row in ensemble_preds_unlabeled]

    # Generar CSV final de submission
    submission = pd.DataFrame({"id": rt_ids, "pred": final_unlabeled})
    submission.to_csv(os.path.join("data", "submission.csv"), index=False)
    print("Archivo de submission generado en data/submission.csv")

if __name__ == "__main__":
    main()
