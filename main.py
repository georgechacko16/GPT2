import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def initialize_model():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a response using the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=model.config.eos_token_id)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    model, tokenizer = initialize_model()

    print("AI: Hello, I'm your conversational AI. You can start the conversation or type 'exit' to quit.")
    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("AI: Goodbye!")
            break

        response = generate_response(user_input, model, tokenizer)
        print("AI:", response)

if __name__ == "__main__":
    main()
