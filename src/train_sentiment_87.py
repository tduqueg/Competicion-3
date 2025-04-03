import os
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
# 0. Configuración
# ============================================================================
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Evita warnings de tokenizers
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

def set_seed(seed=42):
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    """Calcula la métrica de accuracy."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ============================================================================
# 1. Entrenamiento, Evaluación y Submission
# ============================================================================
def main():
    set_seed(42)

    # =========================================================================
    # 1.1 Carga de Dataset IMDB
    # =========================================================================
    imdb = load_dataset("imdb")
    train_ds, test_ds = imdb["train"], imdb["test"]

    # =========================================================================
    # 1.2 Definición de Tokenizador y Modelo
    # =========================================================================
    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # =========================================================================
    # 1.3 Tokenización y Formateo
    # =========================================================================
    def tokenize(ex):
        return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=512)

    train_ds = (train_ds
                .map(tokenize, batched=True)
                .rename_column("label", "labels")
                .remove_columns(["text"]))
    test_ds = (test_ds
               .map(tokenize, batched=True)
               .rename_column("label", "labels")
               .remove_columns(["text"]))

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    # =========================================================================
    # 1.4 Configuración de Entrenamiento
    # =========================================================================
    args = TrainingArguments(
        output_dir="./model_output",
        num_train_epochs=12,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,
        warmup_ratio=0.1,
        weight_decay=0.01
    )

    # =========================================================================
    # 1.5 Entrenamiento 
    # =========================================================================
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # =========================================================================
    # 1.6 Evaluación en Test de IMDB
    # =========================================================================
    res_imdb = trainer.evaluate(test_ds)
    print("Accuracy en test de IMDB:", res_imdb["eval_accuracy"])

    # =========================================================================
    # 1.7 Evaluación 
    # =========================================================================
    rt_labeled = pd.read_csv(os.path.join("data", "rt_test_with_labels.csv"))
    ds_rt_labeled = Dataset.from_pandas(rt_labeled)
    ds_rt_labeled = (ds_rt_labeled
                     .map(tokenize, batched=True)
                     .rename_column("pred", "labels"))
    ds_rt_labeled = ds_rt_labeled.remove_columns(
        [c for c in ["text", "id", "pred"] if c in ds_rt_labeled.column_names]
    )
    ds_rt_labeled.set_format("torch")

    preds_rt = trainer.predict(ds_rt_labeled).predictions
    acc_rt = accuracy_score(rt_labeled["pred"], np.argmax(preds_rt, axis=1))
    print("Accuracy en Rotten Tomatoes (con labels):", acc_rt)

    # =========================================================================
    # 1.8 Predicción 
    # =========================================================================
    rt_unlabeled = pd.read_csv(os.path.join("data", "rt_test_unlabeled.csv"))
    ds_rt_unlabeled = Dataset.from_pandas(rt_unlabeled)
    ds_rt_unlabeled = ds_rt_unlabeled.map(tokenize, batched=True)
    ids = ds_rt_unlabeled["id"]
    ds_rt_unlabeled = ds_rt_unlabeled.remove_columns(
        [c for c in ["text", "id"] if c in ds_rt_unlabeled.column_names]
    )
    ds_rt_unlabeled.set_format("torch")

    preds_unlabeled = trainer.predict(ds_rt_unlabeled).predictions
    submission = pd.DataFrame({"id": ids, "pred": np.argmax(preds_unlabeled, axis=1)})
    submission.to_csv(os.path.join("data", "submission.csv"), index=False)
    print("Archivo de submission generado en data/submission.csv")

if __name__ == "__main__":
    main()
