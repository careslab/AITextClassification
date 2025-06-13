import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch

# 1. Charger et préparer les données
# Charger le fichier CSV avec les données
df = pd.read_csv("dataset_medium.csv")  # Remplace le chemin par le bon fichier CSV

# Encoder les labels (catégories) en entiers
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

# Convertir le DataFrame en dataset Hugging Face
dataset = Dataset.from_pandas(df[["text", "label_id"]])

# Diviser les données en train et test (90% train, 10% test)
dataset = dataset.train_test_split(test_size=0.1)

# 2. Tokenisation des textes
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Fonction de tokenisation des textes
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Appliquer la tokenisation sur le dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# 3. Charger le modèle pré-entraîné DistilBERT pour la classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)  # Nombre de classes à prédire
)

# 4. Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",  # Où enregistrer les résultats
    evaluation_strategy="epoch",  # Évaluer après chaque époque
    logging_strategy="epoch",  # Journaliser après chaque époque
    save_strategy="epoch",  # Sauvegarder après chaque époque
    num_train_epochs=5,  # Nombre d'époques
    per_device_train_batch_size=16,  # Taille des lots d'entraînement
    per_device_eval_batch_size=16,  # Taille des lots pour l'évaluation
    learning_rate=2e-5,  # Taux d'apprentissage
    weight_decay=0.01,  # Décroissance du poids pour éviter le surapprentissage
    load_best_model_at_end=True,  # Charger le meilleur modèle à la fin de l'entraînement
    metric_for_best_model="accuracy",  # Utiliser la précision comme critère pour le meilleur modèle
)

# 5. Calcul des métriques d'évaluation (précision)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  # Prédictions finales en classes
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
model.save_pretrained("./my_finetuned_model")  # Enregistrer le modèle fine-tuné
tokenizer.save_pretrained("./my_finetuned_model")  # Enregistrer le tokenizer
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)  # Sauvegarder l'encodeur de labels pour une utilisation future