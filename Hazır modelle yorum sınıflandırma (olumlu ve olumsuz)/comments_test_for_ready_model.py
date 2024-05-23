import cv2
import pandas as pd
import pytesseract
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Hazır eğitilmiş veri ile kontol edildi
tokenizer = AutoTokenizer.from_pretrained("gurkan08/turkish-product-comment-sentiment-classification")
model = AutoModelForSequenceClassification.from_pretrained("gurkan08/turkish-product-comment-sentiment-classification", num_labels=2)

# # Önceden eğitilmiş distilBERT modelini ve tokenizer'ını yükle
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Etiketler tanımlanıyor
label_contents = {
    'Olumlu': 0,
    'Olumsuz': 1,
}

def estimate_from_text_with_distilBert(text):
     # Modelin giriş formatına dönüştür
    inputs = tokenizer(text, truncation=True, padding='max_length', return_tensors="pt")
    
    # Tahmini yap
    outputs = model(**inputs)
     
    # Tahmin edilen sınıfın indeksini al
    predicted_class_idx = outputs.logits.argmax().item()

    #adı
    predicted_class = list(label_contents.keys())[predicted_class_idx]
    
    # Tahmin edilen sınıfın olasılığını al
    predicted_probability = torch.softmax(outputs.logits, dim=1)[0][predicted_class_idx].item()

    
    # Sonucu göster
    print("Tahmin edilen sınıf:", predicted_class) 
    print("Tahmin edilen sınıfın olasılığı:", predicted_probability)
    
    return predicted_class,predicted_probability


estimate_from_text_with_distilBert("Ben bu ürünü gayet beğendim fakat biraz uzun oldu.")