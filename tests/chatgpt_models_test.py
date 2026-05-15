"""
Quick smoke-test for the OpenAI Chat Completions API.

Usage (OpenAI):
    python tests/test_chatgpt_models.py --api_key sk-... --model gpt-4o-mini

Usage (local vLLM server):
    python tests/test_chatgpt_models.py \
        --base_url http://localhost:8001/v1 \
        --model mistralai/Ministral-3-3B-Reasoning-2512 \
        --api_key EMPTY

The api_key can also be supplied via the OPENAI_API_KEY environment variable.
"""

import argparse
import os
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible chat completion")
    parser.add_argument("--api_key", default=None, help="API key (falls back to OPENAI_API_KEY env var; use any string for vLLM)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model ID, e.g. gpt-4o or mistralai/Ministral-3-3B-Reasoning-2512")
    parser.add_argument("--base_url", default=None, help="Base URL for vLLM or other OpenAI-compatible server, e.g. http://localhost:8001/v1")
    parser.add_argument("--prompt", default="Say hello in one sentence.", help="Prompt to send")
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=args.base_url)

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
