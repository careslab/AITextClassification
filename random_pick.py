import csv
import random

input_file = "./data/NV_command.csv"
output_file = "./data/2000_random_NV.csv"
nb_lignes = 2000

with open(input_file, "r", encoding="utf-8") as fin:
    reader = list(csv.reader(fin))
    header = None
    # Si le fichier a un header, décommente la ligne suivante :
    # header, reader = reader[0], reader[1:]
    if len(reader) < nb_lignes:
        print(f"❌ Le fichier ne contient que {len(reader)} lignes.")
        exit(1)
    sample = random.sample(reader, nb_lignes)

with open(output_file, "w", encoding="utf-8", newline='') as fout:
    writer = csv.writer(fout)
    # Si tu veux garder le header, décommente la ligne suivante :
    # writer.writerow(header)
    writer.writerows(sample)

print(f"✅ {nb_lignes} lignes aléatoires ont été enregistrées dans {output_file}")