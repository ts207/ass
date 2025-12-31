from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-5.2",
    input="Say hello in one sentence."
)
print(resp.output_text)
