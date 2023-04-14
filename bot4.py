
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(
        input_ids=input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        num_return_sequences=1,
    )
    bot_response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response

# Example usage
while True:
    user_input = input("You: ")
    bot_response = generate_response(user_input)
    print("Bot:", bot_response)