import tkinter as tk
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Önceden eğitilmiş distilBERT modelini ve tokenizer'ını yükle
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('./DistilBert/model', num_labels=3)

# Etiketler tanımlanıyor
label_contents = {
    'Olumsuz': 0,
    'Olumlu': 1,
    'Tarafsız': 2,
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




def process_text():
    input_text = entry.get() 
    # text alındı
    label,score = estimate_from_text_with_distilBert(input_text)
    processed_text= str(label)+"  " +str(score)
    result_label.config(text=processed_text)

# Ana pencereyi oluştur
root = tk.Tk()
root.title("Comment Analyser")

# Metin girişi için bir giriş alanı oluştur
entry = tk.Entry(root, width=150)
entry.pack(pady=10)

# Buton oluştur ve işlevini belirle
button = tk.Button(root, text="Send", command=process_text)
button.pack(pady=10)

# Sonucu göstermek için bir etiket oluştur
result_label = tk.Label(root, text="", wraplength=500)
result_label.pack(pady=10)

# Ana döngüyü başlat
root.mainloop()
