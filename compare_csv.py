import csv

train_file = input("Chemin du fichier d'entrainement : ").strip()
test_file = input("Chemin du fichier de test : ").strip()

def read_csv_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(tuple(row) for row in csv.reader(f))

lines1 = read_csv_lines(train_file)
lines2 = read_csv_lines(test_file)

common_lines = lines1 & lines2

if common_lines:
    print(f"✅ Il y a {len(common_lines)} ligne(s) similaire(s) entre les deux fichiers.")
else:
    print("❌ Les fichiers sont différents, aucune ligne similaire trouvée.")