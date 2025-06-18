import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch

# 1. Charger et préparer les données
# Charger le fichier CSV avec les données
df = pd.read_csv("./data/60k_voice_command.csv")  # Remplace le chemin par le bon fichier CSV

# Encoder les labels (catégories) en entiers
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Convertir le DataFrame en dataset Hugging Face
dataset = Dataset.from_pandas(df[["text", "label"]])

# Diviser les données en train et test (90% train, 10% test)
dataset = dataset.train_test_split(test_size=0.1)

# 2. Tokenisation des textes
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

# Fonction de tokenisation des textes
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Appliquer la tokenisation sur le dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# 3. Charger le modèle pour la classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=len(label_encoder.classes_)  
)

# 4. Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",  
    evaluation_strategy="epoch",  
    logging_strategy="epoch", 
    save_strategy="epoch",  
    num_train_epochs=5, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,  
    learning_rate=2e-5,  
    weight_decay=0.01,  
    load_best_model_at_end=True,  
    metric_for_best_model="accuracy",  
)

# 5. Calcul des métriques d'évaluation (précision)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  
    return {"accuracy": accuracy_score(labels, preds)}

# 6. Créer l'objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. Lancer l'entraînement du modèle
trainer.train()

# 8. Sauvegarder le modèle fine-tuné
model.save_pretrained("./60k_finetuned_model")  
tokenizer.save_pretrained("./60k_finetuned_model") 
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)  