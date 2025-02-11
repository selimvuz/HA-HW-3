from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modeli yükle
model = AutoModelForCausalLM.from_pretrained("./finetuned_gpt_large_v3")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_gpt_large_v3")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text_generator = pipeline('text-generation', model=model,
                          tokenizer=tokenizer, max_new_tokens=256)

def get_model_response(instruction):
    instruction_prompt = f"{instruction}"
    result = text_generator(instruction_prompt)
    generated_response = result[0]['generated_text']
    return generated_response[len(instruction_prompt):]

model_response = get_model_response(
    "Bu sunum ne zaman bitecek?")
print(model_response)
