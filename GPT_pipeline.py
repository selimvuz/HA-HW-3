from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modeli yükle
model = AutoModelForCausalLM.from_pretrained("./finetuned_gpt")
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Bir soru sor
question = "Nasılsın?"
answer = pipe(question)

print(answer)
