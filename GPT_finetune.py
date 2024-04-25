import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2-large")
model = AutoModelForCausalLM.from_pretrained(
    "ytu-ce-cosmos/turkish-gpt2-large", return_dict=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# CSV dosyasını yükle ve sütunları birleştir
data = pd.read_csv("datasets/instructions_demo.csv")
data['question'] = data['talimat'] + " " + \
    data['giriş'] + tokenizer.eos_token + data['çıktı']
data = data.dropna(subset=['question', 'çıktı'])  # Eksik değerleri temizle

# Verilerinizi eğitim ve doğrulama setleri olarak ayırma
train_data, eval_data = train_test_split(data, test_size=0.1)

# Hugging Face Dataset nesnelerine dönüştürme
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

# Eğitim ve değerlendirme sırasında verileri de GPU'ya taşı


def tokenize_function(examples):
    combined_text = [(q + " " + g + tokenizer.eos_token + c) for q, g,
                     c in zip(examples['talimat'], examples['giriş'], examples['çıktı'])]
    inputs = tokenizer(combined_text, truncation=True,
                       max_length=512, padding="max_length", return_tensors="pt")
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


tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
    tokenizer=tokenizer
)   

trainer.train()

model.save_pretrained("./finetuned_gpt_large")
