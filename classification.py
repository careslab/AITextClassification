class Classification:
    def __init__(self):

    def tools(self,command):
        model_path = "chemin/vers/ton_modele"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        id2label = dict(enumerate(label_encoder.classes_))
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        prompt = command
        result = classifier(prompt, top_k=1)[0]
        label_id = int(result['label'].split("_")[-1]) if result['label'].startswith("LABEL_") else int(result['label'])
        predicted_label = id2label[label_id]
        print(f"Commande prédite : {predicted_label}")