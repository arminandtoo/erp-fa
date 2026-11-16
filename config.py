import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).parent / ".env")

AVALAI_BASE_URL = "https://api.avalai.ir/v1"
MODEL = os.getenv("MODEL", "gpt-4o-mini")


def get_llm():
    api_key = os.getenv("AVALAI_API_KEY")
    if not api_key:
        raise RuntimeError("AVALAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return ChatOpenAI(model=MODEL, temperature=0.2, api_key=api_key, base_url=AVALAI_BASE_URL)
