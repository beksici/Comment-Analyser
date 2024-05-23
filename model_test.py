import chardet
import pandas as pd
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


#estimate_from_text_with_distilBert("merhaba")



def get_data_from_csv(path,fileName):

    # Dosyanın kodlamasını tespit et
    with open(path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        print(f"Tespit edilen kodlama: {encoding}")

    # CSV dosyasını oku
    df = pd.read_csv(path, encoding=encoding)
    df['DistilBert Result']=""
    df['DistilBert Score']=""
    df['Is it true']=""

    a=0
    # Satır satır işlemek için iterrows() kullan
    for index, row in df.iterrows():
        a+=1
        print(a)
        statu = row['Durum']
        result,score=estimate_from_text_with_distilBert(statu)
        df.at[index,'DistilBert Result']=result
        df.at[index,'DistilBert Score']=score
        if str(statu)==str(result):
             df.at[index,'Is it true']=1
        else:
             df.at[index,'Is it true']=0

    output_excel_path = f"{fileName}.xlsx"
    df.to_excel(output_excel_path, index=False)




excel_path="C:/Users/lenovo/Desktop/Comment Analyser/magaza_yorumlari_duygu_analizi/magaza_yorumlari_duygu_analizi.csv"
get_data_from_csv(excel_path,"analysed")
