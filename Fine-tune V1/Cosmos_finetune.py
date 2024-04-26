import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Tokenizer ve modeli yüklemek
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2-large")
model = AutoModelForCausalLM.from_pretrained("ytu-ce-cosmos/turkish-gpt2-large", return_dict=True)

# Pad token eksikse, onu eos token olarak ayarla
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Cihazı ayarla (GPU veya CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# CSV dosyasını yükle ve gereken sütunları birleştir
data = pd.read_csv("../datasets/soru_cevap.csv")
data['input'] = data['soru'] + tokenizer.eos_token + data['makine_cevabı']
data = data.dropna(subset=['input', 'makine_cevabı'])  # Eksik değerleri temizle

# Verileri eğitim ve doğrulama setleri olarak ayır
train_data, eval_data = train_test_split(data, test_size=0.1)

# Hugging Face Dataset nesnelerine dönüştür
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

# Veri tokenizasyon fonksiyonu
def tokenize_function(examples):
    # Sütun isimlerini yeni veri setinize göre güncelleyin
    inputs = tokenizer(examples['input'], truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    inputs['labels'] = inputs.input_ids.detach().clone()
    eos_mask = inputs.input_ids == tokenizer.eos_token_id
    for i in range(inputs.input_ids.shape[0]):
        eos_indices = (eos_mask[i] == True).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            first_eos_idx = eos_indices[0]
            inputs['labels'][i][:first_eos_idx+1] = -100
        else:
            inputs['labels'][i][:] = -100
    return inputs

# Tokenizasyon işlemi
tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)

# Eğitim argümanları
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    fp16=True,
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=5,
    load_best_model_at_end=True,
)

# Trainer nesnesi
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
    tokenizer=tokenizer
)

# Model eğitimi
trainer.train()

# Modeli kaydet
model.save_pretrained("./finetuned_gpt_large_v1")
tokenizer.save_pretrained("./finetuned_gpt_large_v1")