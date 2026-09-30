# uv run --env-file .env python test_api.py - asi carga el .env

from openai import OpenAI

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Di exactamente: hola"}],
)

print(resp.choices[0].message.content)