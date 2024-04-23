import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "ytu-ce-cosmos/turkish-gpt2", return_dict=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# CSV dosyasını yükle ve sütunları birleştir
data = pd.read_csv("datasets/instructions.csv")
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
    # Prepare the inputs and labels
    inputs = tokenizer(
        examples['question'], truncation=True, max_length=512, padding="max_length")
    inputs = {k: torch.tensor(v).to(device)
              for k, v in inputs.items()}  # GPU'ya taşı
    inputs['labels'] = inputs['input_ids'].detach().clone()
    return inputs


tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    gradient_accumulation_steps=2,  # ?
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    fp16=True,  # Enable mixed precision ?
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
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

model.save_pretrained("./finetuned_gpt")
