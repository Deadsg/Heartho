from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load once for efficiency
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").eval()

def supervised_style_classifier(input_data: str) -> str:
    inputs = bert_tokenizer(input_data, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    label = "positive" if predicted_class == 1 else "negative"
    return label
