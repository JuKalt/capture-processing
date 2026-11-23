"""Interactive Hugging Face login helper for gated processing models."""

from __future__ import annotations

import getpass
import os
import sys


SEMANTIC_MODEL_URL = "https://huggingface.co/facebook/sapiens2-seg-1b"
MODEL_URL = "https://huggingface.co/facebook/sam-3d-body-dinov3"
LOGIN_URL = "https://huggingface.co/login"
TOKEN_URL = "https://huggingface.co/settings/tokens"
MAX_ATTEMPTS = 3


def token_format_ok(token):
    return bool(token and token.isascii() and all(33 <= ord(c) < 127 for c in token))


def main() -> None:
    try:
        from huggingface_hub import login, whoami
        from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError
    except ImportError as exc:
        raise SystemExit(
            "MHR needs `huggingface_hub` in the MHR Python environment. "
            "Install it or choose another environment with --mhr-python."
        ) from exc

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        user = None if env_token and not token_format_ok(env_token) else whoami(token=env_token)
    except LocalTokenNotFoundError:
        user = None
    except HfHubHTTPError as exc:
        if exc.response.status_code not in (401, 403):
            raise
        user = None


    if user:
        name = user.get("name") or user.get("fullname") or "authenticated user"
        print(f"Hugging Face authentication found for: {name}")
        return

    if env_token:
        raise SystemExit(
            "The Hugging Face token in the environment could not be validated. "
            "Replace it or unset HF_TOKEN and HUGGING_FACE_HUB_TOKEN to sign in interactively."
        )

    if not sys.stdin.isatty():
        raise SystemExit(
            "The processing model downloads need authorized Hugging Face access.\n"
            f"  Request/access page: {MODEL_URL}\n"
            f"  Semantic model access: {SEMANTIC_MODEL_URL}\n"
            f"  Login page: {LOGIN_URL}\n"
            f"  Token page: {TOKEN_URL}\n"
            "Run the installer interactively or provide an authorized HF_TOKEN."
        )

    print("The processing model downloads need authorized Hugging Face access.")
    print(f"  Request/access page: {MODEL_URL}")
    print(f"  Semantic model access: {SEMANTIC_MODEL_URL}")
    print(f"  Login page: {LOGIN_URL}")
    print(f"  Token page: {TOKEN_URL}")
    for attempt in range(MAX_ATTEMPTS):
        token = getpass.getpass("Hugging Face read token: ").strip()
        if token_format_ok(token):
            try:
                user = whoami(token=token)
            except HfHubHTTPError as exc:
                if exc.response.status_code not in (401, 403):
                    raise
            else:
                login(token=token, add_to_git_credential=False)
                name = user.get("name") or user.get("fullname") or "authenticated user"
                print(f"Hugging Face authentication saved for: {name}")
                return
        remaining = MAX_ATTEMPTS - attempt - 1
        if remaining:
            print(f"Hugging Face rejected the token. Try again ({remaining} attempts left).",
                  file=sys.stderr, flush=True)
    raise SystemExit(f"Hugging Face sign-in failed after {MAX_ATTEMPTS} attempts. Check {TOKEN_URL}.")


if __name__ == "__main__":
    main()
