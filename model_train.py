import chardet
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Önceden eğitilmiş distilBERT modelini ve tokenizer'ını yükle
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Etiketler tanımlanıyor
label_contents = {
    'Olumsuz': 0,
    'Olumlu': 1,
    'Tarafsız': 2,
}

# data bulunduğu klasör yolu
data_path = 'C:/Users/lenovo/Desktop/Comment Analyser/magaza_yorumlari_duygu_analizi'

# Veri setini oluştur
veri_seti = []

def get_data_from_excel(path):

    # Dosyanın kodlamasını tespit et
    with open(path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        print(f"Tespit edilen kodlama: {encoding}")

    # CSV dosyasını oku
    df = pd.read_csv(path, encoding=encoding)

    a=0
    # Satır satır işlemek için iterrows() kullan
    for index, row in df.iterrows():
        a+=1
        print(a)
        comment = row['Görüş']
        statu = row['Durum']
        veri_seti.append({'text': str(comment), 'label': label_contents[statu]})


excel_path="C:/Users/lenovo/Desktop/Comment Analyser/magaza_yorumlari_duygu_analizi/magaza_yorumlari_duygu_analizi.csv"
get_data_from_excel(excel_path)

print(veri_seti[0])
print(len(veri_seti))
# veri_seti = random.sample(veri_seti, 3000)
# print(len(veri_seti))


# Veri setini eğitim ve test olarak böl
train_set, test_set = train_test_split(veri_seti, test_size=0.2)

# Huggingface datasets formatına dönüştür
train_dataset = Dataset.from_pandas(pd.DataFrame(train_set))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_set))

# DatasetDict oluştur
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Eğitim argümanlarını tanımla
training_args = TrainingArguments(
    output_dir='./DistilBert/results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./DistilBert/logs',
    logging_steps=10,
)


train_encoded = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
test_encoded = test_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=test_encoded
)

# Eğitimi başlat
trainer.train()

# Modeli kaydet
model.save_pretrained('./DistilBert/model')

# Modeli yükle ve kullan
model = DistilBertForSequenceClassification.from_pretrained('./DistilBert/model')

# Yeni bir resim üzerinde tahmin yap
# ...

# Tahminleri hesapla
predictions = trainer.predict(test_encoded)

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(predictions.label_ids, predictions.predictions.argmax(axis=1))
# Sınıflandırma raporunu hesapla
report = classification_report(predictions.label_ids, predictions.predictions.argmax(axis=1))

# Hassasiyet hesapla
precision = precision_score(predictions.label_ids, predictions.predictions.argmax(axis=1), average='weighted')

# Duyarlılık hesapla
recall = recall_score(predictions.label_ids, predictions.predictions.argmax(axis=1), average='weighted')

# F1 puanı hesapla
f1 = f1_score(predictions.label_ids, predictions.predictions.argmax(axis=1), average='weighted')

print("Model doğruluğu:", accuracy)
print("Hassasiyet:", precision)
print("Duyarlılık:", recall)
print("F1 Puanı:", f1)
print("Sınıflandırma Raporu:")
print(report)

print(predictions.label_ids)
print(predictions.predictions)

for i in range(len(predictions.label_ids)):
    print("Tahmin edilen sınıf:", predictions.label_ids[i])
    print("Tahmini olasılıklar:")
    for j, score in enumerate(predictions.predictions[i]):
        print(f"Sınıf {j}: {score}")
    print()

