from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-4.1"
PROMPT = "In one sentence, explain why grounded citations improve enterprise AI reliability."


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(f"OPENAI_API_KEY is missing from {PROJECT_ROOT / '.env'}", file=sys.stderr)
        return 1

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=30,
    )

    try:
        response = client.responses.create(model=MODEL, input=PROMPT)
    except Exception as exc:
        print(f"OpenAI API test failed: {exc}", file=sys.stderr)
        return 1

    print(f"Model: {MODEL}")
    print(f"Response: {response.output_text.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
