from transformers import AutoTokenizer, MistralForCausalLM

model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def hf_mistral(question, n):
    prompt = question
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=n)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]