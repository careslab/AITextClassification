import csv

input_file = "./data/60k_voice_command.csv"

rows = []
with open(input_file, "r", encoding="utf-8") as fin:
    reader = csv.reader(fin)
    for row in reader:
        # Supposons que la phrase est en première colonne et le label en deuxième
        phrase = row[0]
        label = row[1] if len(row) > 1 else ""
        # Si la phrase commence et finit par des guillemets, on les enlève
        if phrase.startswith('"') and phrase.endswith('"'):
            phrase = phrase[1:-1]
        # On enlève les virgules dans la phrase
        phrase = phrase.replace(",", "")
        rows.append([phrase, label])

with open(input_file, "w", encoding="utf-8", newline='') as fout:
    writer = csv.writer(fout)
    writer.writerows(rows)

print(f"✅ Nettoyage terminé : virgules supprimées dans la première colonne et guillemets enlevés dans {input_file}")