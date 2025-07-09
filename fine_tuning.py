import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import os

# Demander le chemin du fichier CSV et le nom du modèle
csv_path = input("Chemin du fichier CSV pour l'entraînement : ").strip()
model_name = input("Nom du modèle HuggingFace à utiliser (ex: distilroberta-base) : ").strip()

# Extraire le nom du fichier sans extension pour le dossier de sauvegarde
csv_base = os.path.splitext(os.path.basename(csv_path))[0]
save_dir = f"{csv_base}_{model_name.replace('/', '_')}"

# 1. Charger et préparer les données
df = pd.read_csv(csv_path)

# Encoder les labels (catégories) en entiers
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Convertir le DataFrame en dataset Hugging Face
dataset = Dataset.from_pandas(df[["text", "label"]])

# Diviser les données en train et test (80% train, 20% test)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

# 2. Tokenisation des textes
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 3. Charger le modèle pour la classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_encoder.classes_)
)

# 4. Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir=save_dir,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=32,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 7. Lancer l'entraînement du modèle
train_output = trainer.train()

# 8. Sauvegarder le modèle fine-tuné et le tokenizer dans le dossier dédié
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print(f"✅ Modèle et tokenizer sauvegardés dans le dossier : {save_dir}")

# Affichage des courbes d'apprentissage
log_history = trainer.state.log_history
epochs = [entry["epoch"] for entry in log_history if "epoch" in entry and "loss" in entry]
train_loss = [entry["loss"] for entry in log_history if "epoch" in entry and "loss" in entry]
eval_acc = [entry["eval_accuracy"] for entry in log_history if "eval_accuracy" in entry]
eval_epochs = [entry["epoch"] for entry in log_history if "eval_accuracy" in entry]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Train Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de perte (Train)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(eval_epochs, eval_acc, label="Eval Accuracy", marker="o", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Précision sur le jeu de validation")
plt.legend()
plt.grid(True)
plt.show()

