"""Environment and OpenAI client configuration."""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file if present
load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_RETRIES = 3

# input, output cost per 1M tokens
PRICING = {
    "gpt-4.1": (1.00, 3.00),
}

__all__ = ["client", "MODEL", "TEMPERATURE", "MAX_RETRIES", "PRICING"]
