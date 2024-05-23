
# Comment-Analyser
TR

Bu model 11479 türkçe yorumla ve DistilBERT model kullanarak eğitildi.
4253 positive
4238 negative
2938 nötr

EN

This model was trained using 11479 Turkish comments and the DistilBERT model.
4253 positive
4238 negative
2938 nötr




## Usage
### import
    import pandas as pd
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import torch

#### Önceden eğitilmiş distilBERT modelini ve tokenizer'ını yükle
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('./DistilBert/model', num_labels=3)

#### Etiketleri tanımla

    label_contents = {
    'Olumsuz': 0,
    'Olumlu': 1,
    'Tarafsız': 2,
}


#### Fonksiyon
    def estimate_from_text_with_distilBert(text):
#### Modelin giriş formatına dönüştür

    inputs = tokenizer(text, truncation=True,padding='max_length', return_tensors="pt")

#### Tahmini yap

    outputs = model(**inputs)
 
#### Tahmin edilen sınıfın indeksini al

    predicted_class_idx = outputs.logits.argmax().item()

#### sınıfı al

    predicted_class = list(label_contents.keys())[predicted_class_idx]

#### Tahmin edilen sınıfın olasılığını al

    predicted_probability = torch.softmax(outputs.logits, dim=1)[0][predicted_class_idx].item()


#### Sonucu göster

    print("Tahmin edilen sınıf:", predicted_class) 
    print("Tahmin edilen sınıfın olasılığı:", predicted_probability)

    return predicted_class,predicted_probability



    
## Results


![trainin time](https://github.com/beksici/Comment-Analyser/assets/136181100/73b8bf74-490b-406a-b56f-eebc3e99d59d)


![result-2](https://github.com/beksici/Comment-Analyser/assets/136181100/2b3968f6-4389-4828-8333-fc761fdba5db)


![result-1](https://github.com/beksici/Comment-Analyser/assets/136181100/90256f1c-67f0-43a9-a822-0f49df2e8860)
