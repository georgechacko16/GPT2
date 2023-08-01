import openai

openai.api_key = "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"

def get_gpt3_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # This specifies the language model to use
        prompt=prompt,
        max_tokens=150,  # The maximum number of tokens in the response
        temperature=0.7,  # Controls the randomness of the output. Higher values make it more random, lower values make it more focused.
        stop=None,  # You can provide a list of stop sequences to stop the model from generating after encountering them.
    )
    return response.choices[0].text.strip()

while True:
    prompt: str = input("YOU: ")

    if prompt == "quit": break

    get_gpt3_response(prompt)