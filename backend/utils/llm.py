import os
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_response(user_input: str, context: dict = None) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Cerebrum, a memory-aware assistant."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=700
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"
