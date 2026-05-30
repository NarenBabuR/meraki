"""Quick check that ANTHROPIC_API_KEY (from .env) is valid.

    python scripts/test_key.py

Makes one tiny, cheap Claude call and reports a clear pass/fail. Never prints
the key itself — only its length and prefix so you can sanity-check the shape.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import anthropic  # noqa: E402

MODEL = os.getenv("GEN_MODEL", "claude-haiku-4-5-20251001")


def main() -> int:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("FAIL: ANTHROPIC_API_KEY not found in environment or .env")
        return 1

    # Shape check only — never print the secret.
    print(f"Key loaded: length={len(key)} prefix={key[:10]!r}")
    if not key.startswith("sk-ant-"):
        print("WARNING: key does not start with 'sk-ant-' — likely not a Console API key.")

    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        print(f"PASS: {MODEL} responded -> {text!r}")
        print(f"  tokens: in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        return 0
    except anthropic.AuthenticationError as e:
        print(f"FAIL (401 auth): the key was rejected -> {e}")
        print("  -> Use a Console API key from console.anthropic.com (Settings > API Keys).")
        return 1
    except anthropic.NotFoundError as e:
        print(f"FAIL (404 model): '{MODEL}' not available to this key -> {e}")
        print("  -> Try: GEN_MODEL=claude-haiku-4-5-20251001 python scripts/test_key.py")
        return 1
    except anthropic.APIStatusError as e:
        print(f"FAIL ({e.status_code}): {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL ({type(e).__name__}): {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
