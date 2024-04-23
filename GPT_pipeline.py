from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modeli yükle
model = AutoModelForCausalLM.from_pretrained("./finetuned_gpt")
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text_generator = pipeline('text-generation', model=model,
                          tokenizer=tokenizer, max_new_tokens=256)

def get_model_response(instruction):
    instruction_prompt = f"### Kullanıcı:\n{instruction}\n### Asistan:\n"
    result = text_generator(instruction_prompt)
    generated_response = result[0]['generated_text']
    return generated_response[len(instruction_prompt):]

model_response = get_model_response(
    "Makarna nasıl yapılır açıklar mısın?")
print(model_response)
