import os
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

class GroqLLM:
    """
    Simple wrapper so our RAG agents can call Groq like GroqLLM.
    """

    def __init__(self, model="llama-3.3-70b-versatile"):
        load_dotenv()
        self.model = model
        groq_api = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=groq_api)
    
    def invoke(self, messages):
        """
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq LLM failed: {str(e)}")

    def stream(self, messages):
        """Streaming generator"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"Groq streaming failed: {str(e)}")
