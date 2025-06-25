import csv
import re

input_file = "./data/NV_command.csv"

def clean_text(text):
    # Garde uniquement les lettres (majuscules/minuscules), les espaces et les virgules
    return re.sub(r"[^a-zA-Z, ]", "", text)

rows = []
with open(input_file, "r", encoding="utf-8") as fin:
    reader = csv.reader(fin)
    for row in reader:
        # Nettoie chaque colonne
        cleaned_row = [clean_text(col) for col in row]
        rows.append(cleaned_row)

with open(input_file, "w", encoding="utf-8", newline='') as fout:
    writer = csv.writer(fout)
    writer.writerows(rows)

print(f"✅ Nettoyage terminé : seuls les lettres, espaces et virgules sont conservés dans {input_file}")