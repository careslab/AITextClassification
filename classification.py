from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle

class Classification:
    def __init__(self):


    def sort(self, command):
        #load the model and tokenizer
        model_path = "./data/w_cat/sort/sort_distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #load the label encoder
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        id2label = dict(enumerate(label_encoder.classes_))
        #create the classifier pipeline
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        #classify the command
        prompt = command
        result = classifier(prompt, top_k=1)[0]
        label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
        predicted_label = id2label[label_id]
        #print the predicted label
        print(f"Commande prédite : {predicted_label}")


    def tools(self,command):
        #load the model and tokenizer
        model_path = "./data/w_cat/tools/tools_distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #load the label encoder
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        id2label = dict(enumerate(label_encoder.classes_))
        #create the classifier pipeline
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        #classify the command
        prompt = command
        result = classifier(prompt, top_k=1)[0]
        label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
        predicted_label = id2label[label_id]
        #print the predicted label
        print(f"Commande prédite : {predicted_label}")

    def start_stop(self, command):
        #load the model and tokenizer
        model_path = "./data/w_cat/start_stop/start_stop_distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #load the label encoder
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        id2label = dict(enumerate(label_encoder.classes_))
        #create the classifier pipeline
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        #classify the command
        prompt = command
        result = classifier(prompt, top_k=1)[0]
        label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
        predicted_label = id2label[label_id]
        #print the predicted label
        print(f"Commande prédite : {predicted_label}")

    def camera(self,command):
        #load the model and tokenizer
        model_path = "./data/w_cat/camera/camera_distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #load the label encoder
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        id2label = dict(enumerate(label_encoder.classes_))
        #create the classifier pipeline
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        #classify the command
        prompt = command
        result = classifier(prompt, top_k=1)[0]
        label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
        predicted_label = id2label[label_id]
        #print the predicted label
        print(f"Commande prédite : {predicted_label}")

    def draw(self,command):
        #load the model and tokenizer
        model_path = "./data/w_cat/draw/draw_distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #load the label encoder
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        id2label = dict(enumerate(label_encoder.classes_))
        #create the classifier pipeline
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        #classify the command
        prompt = command
        result = classifier(prompt, top_k=1)[0]
        label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
        predicted_label = id2label[label_id]
        #print the predicted label
        print(f"Commande prédite : {predicted_label}")

    def patient(self,command):
        #load the model and tokenizer
        model_path = "./data/w_cat/patient/patient_distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #load the label encoder
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        id2label = dict(enumerate(label_encoder.classes_))
        #create the classifier pipeline
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        #classify the command
        prompt = command
        result = classifier(prompt, top_k=1)[0]
        label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
        predicted_label = id2label[label_id]
        #print the predicted label
        print(f"Commande prédite : {predicted_label}")


