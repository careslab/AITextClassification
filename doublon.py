import pandas as pd
df = pd.read_csv("./data/60k_voice_command.csv")
print("Avant:", len(df))
df = df.drop_duplicates(subset="text")
print("Après:", len(df))
