import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chat_with_model(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    with torch.no_grad():
        response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    print("AI: Hello! I'm your AI chatbot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("AI: Goodbye!")
            break
        response = chat_with_model(user_input)
        print("AI:", response)

#at the end, model26 chatBot creates some random response.
