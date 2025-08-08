
from classification import Classification
import pandas as pd
import time
import csv
import os

clf = Classification()

# Demande le chemin du fichier CSV à l'utilisateur
csv_file_path = input("Chemin du fichier CSV à tester : ").strip()

def load_list_from_csv(csv_path, size=None):
    df = pd.read_csv(csv_path, header=None)
    if size is not None:
        df = df.sample(n=size)
    prompt_list = df[0].tolist()
    response_list = df[1].tolist()
    return prompt_list, response_list

prompt_list, response_list = load_list_from_csv(csv_file_path)

if len(prompt_list) != len(response_list):
    raise ValueError("The prompt and response lists must be the same length.")

# Demande le nombre de tests
nb_of_test = int(input("How many test do you want to do on this model ?\n"))

# Récupère le dossier parent et le nom du dossier du fichier CSV
csv_dir = os.path.dirname(os.path.abspath(csv_file_path))
csv_folder_name = os.path.basename(csv_dir)

# Utilise le nom du dossier pour nommer les fichiers de résultats
csv_filename = os.path.join(csv_dir, f"{csv_folder_name}_results.csv")
incorrect_filename = os.path.join(csv_dir, f"{csv_folder_name}_incorrect_predictions.csv")

incorrect_file = open(incorrect_filename, mode='w', newline='')
incorrect_writer = csv.writer(incorrect_file)
incorrect_writer.writerow(["Prompt", "Expected", "Answered"])

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Test #", "Average Time (s)", "Accuracy (%)"])  # Header

    total_time_cnt = 0
    total_accuracy_cnt = 0

    for j in range(nb_of_test):
        time_cnt = 0
        accuracy_cnt = 0

        for i in range(len(prompt_list)):
            prompt = prompt_list[i]
            expected = str(response_list[i]).strip()

            send_time = time.time()
            response = clf.handle_command(prompt)
            receive_time = time.time()

            elapse_time = receive_time - send_time
            time_cnt += elapse_time

            if response == expected:
                accuracy_cnt += 1
            else:
                incorrect_writer.writerow([prompt, expected, response])

        time_avg = time_cnt / len(prompt_list)
        accuracy = (accuracy_cnt * 100) / len(prompt_list)
        total_time_cnt += time_avg
        total_accuracy_cnt += accuracy

        print(f"Iteration : {j+1:.2f}\nAverage response time: {time_avg:.2f}s\nAccuracy: {accuracy:.2f}%\n")
        writer.writerow([j+1, f"{time_avg:.2f}", f"{accuracy:.2f}"])

    total_time_avg = total_time_cnt / nb_of_test
    total_accuracy_avg = total_accuracy_cnt / nb_of_test

    writer.writerow(["TOTAL", f"{total_time_avg:.2f}", f"{total_accuracy_avg:.2f}"])
    print(f"Total average response time: {total_time_avg:.2f}s\nTotal average response accuracy: {total_accuracy_avg:.2f}%\n")

incorrect_file.close()
