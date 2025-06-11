import random
import csv
import pickle
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys

class IterativeFineTuner:
    def __init__ (self):
        self.training_filename = "training_file.csv"
        self.testing_filename = "testing_file.csv"
        self.model_name = None
        self.model_dir = None
        self.learning_rate = None
        self.nb_epochs = None

    def check_file_exists(self, filename):
        """Check if a file exists and is not empty."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"❌ File '{filename}' not found.")
        if os.path.getsize(filename) == 0:
            raise ValueError(f"❌ File '{filename}' is empty.")
        print(f"📂 File '{filename}' is ready ✅")

    def fine_tuning (self):
        # Check if the training file exists
        self.check_file_exists(self.training_filename)

        #read the csv file 
        try:
            df = pd.read_csv(self.training_filename) 
            print("📚 Training CSV file loaded successfully ✅")
        except Exception as e:
            raise ValueError(f"❌ Error reading the training file: {e}")
        
        # Check if required columns exist
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("❌ The training file must contain 'text' and 'label' columns.")

        #encode the labels 
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])
        print("🔑 Labels encoded successfully ✅")
        #convert datafram in dataset
        dataset = Dataset.from_pandas(df[["text", "label"]])
        #split the dataset in train and test
        dataset = dataset.train_test_split(test_size=0.1)
        print("📊 Dataset split into train and test sets ✅")
        #load the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("📝 Tokenizer loaded successfully ✅")
        except Exception as e:
            raise ValueError(f"❌ Error loading tokenizer: {e}")

        def tokenize(batch):
            return tokenizer(batch["text"], padding=True, truncation=True)
        
        #apply the tokenization on the dataset
        tokenized_dataset = dataset.map(tokenize, batched=True)
        print("📝 Dataset tokenized successfully ✅")
        #load the model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(label_encoder.classes_)  
            )
            print("🤖 Model loaded successfully ✅")
        except Exception as e:
            raise ValueError(f"❌ Error loading model: {e}")

        #define the training arguments
        training_args = TrainingArguments(
            output_dir=self.model_dir,  
            evaluation_strategy="epoch",  
            logging_strategy="epoch",  
            save_strategy="epoch",  
            num_train_epochs=self.nb_epochs,  
            per_device_train_batch_size=16,  
            per_device_eval_batch_size=16,  
            learning_rate=self.learning_rate,  
            weight_decay=0.01,  
            load_best_model_at_end=True,  
            metric_for_best_model="accuracy",  
        )
        print("⚙️ Training arguments defined successfully ✅")

        #metrics calculation 
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1) 
            return {"accuracy": accuracy_score(labels, preds)}
        print("📊 Metrics computation function defined ✅")
            
        #create the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        print("👨‍🏫 Trainer created successfully ✅")
        #train the model
        try:
            trainer.train()
            print("🚀 Model training completed successfully ✅")
        except Exception as e:
            raise ValueError(f"❌ Error during training: {e}")

        #save the model / tokenizer / label encoder
        try:
            trainer.save_model(self.model_dir)
            tokenizer.save_pretrained(self.model_dir)
            with open(f"{self.model_dir}/label_encoder.pkl", "wb") as f:
                pickle.dump(label_encoder, f)
            print("💾 Model, tokenizer, and label encoder saved successfully ✅")
        except Exception as e:
            raise ValueError(f"❌ Error saving model or tokenizer: {e}")

    def load_list_from_csv(self , size=100):
        # Check if the testing file exists
        self.check_file_exists(self.testing_filename)

        # Load the testing file
        try:
            df = pd.read_csv(self.testing_filename)
            print("📚 Testing CSV file loaded successfully ✅")
        except Exception as e:
            raise ValueError(f"❌ Error reading the testing file: {e}")

        # Sample the data
        sampled_df = df.sample(n=size)
        prompt_list = sampled_df['text'].tolist()
        response_list = sampled_df['label'].tolist()
        print("🏁 Data ready for testing ✅")
        return prompt_list, response_list

    def test_model(self):
        prompt_list , response_list = self.load_list_from_csv()

        # Ensure the lists are of the same length
        if len(prompt_list) != len(response_list):
            raise ValueError("❌ The prompt and response lists must have the same length.")

        # Load the tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            print("🤖 Model and tokenizer loaded successfully for testing ✅")
        except Exception as e:
            raise ValueError(f"❌ Error loading model or tokenizer for testing: {e}")

        # Load the label encoder
        try:
            with open(f"{self.model_dir}/label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
            print("🔑 Label encoder loaded successfully ✅")
        except Exception as e:
            raise ValueError(f"❌ Error loading label encoder: {e}")

        id2label = dict(enumerate(label_encoder.classes_))
        label2id = {v: k for k, v in id2label.items()}

        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

        def generate_with_transformers(prompt):
            try:
                result = classifier(prompt, top_k=1)[0]
                label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
                return id2label[label_id]
            except Exception as e:
                print(f"❌ Error generating response for prompt '{prompt}': {e}")
                return "UNKNOWN"

        time_cnt = 0
        accuracy_cnt = 0
        incorrect_responses_nb = 0

        # Begin the test loop
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]
            send_time = time.time()
            response = generate_with_transformers(prompt)
            receive_time = time.time()
            print(f"➡️ EXPECTED: {response_list[i]} / RECEIVED: {response}")
            elapse_time = receive_time - send_time 
            time_cnt += elapse_time
            
            if response == response_list[i]:
                accuracy_cnt += 1
                print("✅ Correct response!")
            else:
                incorrect_responses_nb += 1
                print(f"❌ Incorrect response for prompt: {prompt}. Adding it to the training file.")
                with open(self.training_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([prompt, response_list[i]])
                print(f"✏️ Added to training file: {prompt} - {response_list[i]} ✅")

        # Calculating the evaluation metrics
        time_avg = time_cnt / len(prompt_list)
        accuracy = (accuracy_cnt * 100) / len(prompt_list)
        return time_avg, accuracy, incorrect_responses_nb
    
    def run(self):
        iteration_cnt = 1
        accuracy = 0

        while accuracy < 99 :
            if iteration_cnt == 1:
                print("🥇 First-time fine-tuning distilroberta-base")
                self.model_name = "distilroberta-base"
                self.model_dir = f"./{iteration_cnt}_finetuned_roberta"
                self.learning_rate = 2e-5
                self.nb_epochs = 3
            else :
                print(f"👩‍💻 Fine-tuning iteration {iteration_cnt}")
                self.model_name = f"./{iteration_cnt-1}_finetuned_roberta"
                self.model_dir = f"./{iteration_cnt}_finetuned_roberta"
                self.learning_rate = 1e-5
                self.nb_epochs = 2

            self.fine_tuning()
            print("📊 Starting model testing")
            time_avg, accuracy, incorrect_responses_nb = self.test_model()
            print(f"📈 Iteration {iteration_cnt} results:\n ⏰ Average time: {time_avg:.2f}s\n ✅ Accuracy: {accuracy:.2f}%\n ❌ Incorrect responses: {incorrect_responses_nb}")
            iteration_cnt += 1



