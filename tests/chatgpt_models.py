"""
Quick smoke-test for the OpenAI Chat Completions API.

Usage:
    python tests/test_chatgpt_models.py --api_key sk-... --model gpt-4o-mini

The api_key can also be supplied via the OPENAI_API_KEY environment variable;
omit --api_key in that case.
"""

import argparse
import os
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI chat completion")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model ID, e.g. gpt-4o or gpt-4o-mini")
    parser.add_argument("--prompt", default="Say hello in one sentence.", help="Prompt to send")
    parser.add_argument("--max_tokens", type=int, default=32)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    messages = [{"role": "user", "content": args.prompt}]

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
