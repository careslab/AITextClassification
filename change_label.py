import pandas as pd

# Chemin du fichier CSV
csv_path = input("Chemin du fichier CSV : ").strip()
nouveau_label = input("Nouveau label à appliquer à toute la colonne : ").strip()

# Charger le CSV
df = pd.read_csv(csv_path)

# Remplacer toute la colonne 'label' par le nouveau label
df['label'] = nouveau_label

# Sauvegarder dans un nouveau fichier (pour éviter d'écraser l'original)
nouveau_csv = csv_path.replace(".csv", "_label_modifie.csv")
df.to_csv(nouveau_csv, index=False)

print(f"✅ Tous les labels ont été remplacés par '{nouveau_label}' dans {nouveau_csv}")