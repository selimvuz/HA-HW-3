import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline

tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2-large")
model = AutoModelForCausalLM.from_pretrained(
    "ytu-ce-cosmos/turkish-gpt2-large", return_dict=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# CSV dosyasını yükle ve sütunları birleştir
data = pd.read_csv("datasets/instructions.csv")
data['question'] = data['talimat'] + " " + \
    data['giriş'] + tokenizer.eos_token + data['çıktı']
data = data.dropna(subset=['question', 'çıktı'])  # Eksik değerleri temizle

# Hugging Face Dataset nesnesine dönüştür
dataset = Dataset.from_pandas(data)

# Eğitim ve değerlendirme sırasında verileri de GPU'ya taşıyın


def tokenize_function(examples):
    # Prepare the inputs and labels
    inputs = tokenizer(
        examples['question'], truncation=True, max_length=512, padding="max_length")
    inputs = {k: torch.tensor(v).to(device)
              for k, v in inputs.items()}  # GPU'ya taşı
    inputs['labels'] = inputs['input_ids'].detach().clone()
    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    gradient_accumulation_steps=2,  # ?
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True,  # Enable mixed precision ?
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=1000,  # Evaluation and Save happens every 500 steps
    save_total_limit=5,  # Only last 5 models are saved
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./fine_tuned_turkish_gpt2")

# Modeli yükle
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_turkish_gpt2")
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Bir soru sor
question = "Ekonomi ne durumda?"
answer = pipe(question)

print(answer)
