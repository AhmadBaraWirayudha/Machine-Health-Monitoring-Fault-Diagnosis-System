"""
API Authentication Middleware
==============================
Lightweight API key authentication for the FastAPI REST API.

Features:
  - Header-based API key (X-API-Key)
  - Multiple keys supported (for different clients/teams)
  - Key rotation without downtime
  - Public routes (no auth required): /, /health, /docs, /redoc
  - Rate-limit friendly: 401 response with retry guidance

Setup:
    1. Add API keys to .env:
           API_KEY_1=cbm_prod_abc123
           API_KEY_2=cbm_dev_xyz789
       or a comma-separated list:
           API_KEYS=cbm_prod_abc123,cbm_dev_xyz789

    2. Register the middleware in src/api/main.py:
           from src.api.auth import require_api_key, APIKeyMiddleware
           app.add_middleware(APIKeyMiddleware)

    3. Or protect specific routes with the dependency:
           @app.post("/predict")
           async def predict(
               reading: SensorReading,
               _key: str = Depends(require_api_key),
           ):
               ...

Usage — calling the secured API:
    curl -H "X-API-Key: cbm_prod_abc123" http://localhost:8000/predict ...
"""

import os
import hashlib
import logging
import secrets
from pathlib import Path
from typing import Callable

try:
    from fastapi import Request, HTTPException, Depends
    from fastapi.responses import JSONResponse
    from fastapi.security import APIKeyHeader
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    raise ImportError("FastAPI not installed. Run: pip install fastapi")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

log = logging.getLogger(__name__)

# ── Public routes that bypass auth ────────────────────────────────────────────
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json",
                "/favicon.ico", "/nginx-health"}

# ── Load valid API keys ───────────────────────────────────────────────────────

def _load_api_keys() -> set[str]:
    """
    Load valid API keys from environment variables.

    Checks (in order):
      1. API_KEYS=key1,key2,key3   (comma-separated list)
      2. API_KEY_1, API_KEY_2, ... (numbered variables)
      3. API_KEY                   (single key)
    """
    keys: set[str] = set()

    # Multi-key CSV
    csv_val = os.getenv("API_KEYS", "")
    for k in csv_val.split(","):
        k = k.strip()
        if k:
            keys.add(k)

    # Numbered keys API_KEY_1, API_KEY_2, …
    i = 1
    while True:
        k = os.getenv(f"API_KEY_{i}", "").strip()
        if not k:
            break
        keys.add(k)
        i += 1

    # Single key fallback
    single = os.getenv("API_KEY", "").strip()
    if single:
        keys.add(single)

    if not keys:
        # Development mode: auto-generate a key and warn
        dev_key = "cbm_dev_" + secrets.token_hex(8)
        keys.add(dev_key)
        log.warning(
            f"No API keys configured in .env — using auto-generated dev key: {dev_key}\n"
            "Set API_KEYS=your_key in .env to configure production keys."
        )

    # Store hashed copies for logging (never log raw keys)
    for k in keys:
        masked = k[:6] + "***" + k[-3:] if len(k) > 9 else "***"
        log.debug(f"[auth] Loaded API key: {masked}")

    return keys


_VALID_KEYS: set[str] | None = None   # lazy load


def _get_keys() -> set[str]:
    global _VALID_KEYS
    if _VALID_KEYS is None:
        _VALID_KEYS = _load_api_keys()
    return _VALID_KEYS


def reload_keys() -> None:
    """Force reload keys from environment (call after rotating keys)."""
    global _VALID_KEYS
    _VALID_KEYS = _load_api_keys()
    log.info(f"[auth] Reloaded {len(_VALID_KEYS)} API key(s)")


# ── FastAPI dependency ────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    Usage:
        @app.get("/protected")
        async def protected(key: str = Depends(require_api_key)):
            ...
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "Missing API key",
                "message": "Include your API key in the X-API-Key header.",
                "docs":    "https://your-domain.com/docs",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key not in _get_keys():
        log.warning(f"[auth] Invalid API key attempt: {api_key[:4]}***")
        raise HTTPException(
            status_code=403,
            detail={
                "error":   "Invalid API key",
                "message": "The provided API key is not valid or has been revoked.",
            },
        )

    return api_key


# ── Middleware (applies to entire app) ────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that checks X-API-Key on all non-public routes.

    Register in main.py:
        app.add_middleware(APIKeyMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path

        # Always allow public routes
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error":   "Missing API key",
                    "message": "Include your API key in the X-API-Key header.",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if api_key not in _get_keys():
            log.warning(
                f"[auth] Rejected request to {path} with invalid key "
                f"{api_key[:4]}*** from {request.client.host}"
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid or revoked API key."},
            )

        # Inject masked key into request state for logging
        request.state.api_key_prefix = api_key[:6]
        return await call_next(request)


# ── Key management utilities ──────────────────────────────────────────────────

def generate_api_key(prefix: str = "cbm") -> str:
    """
    Generate a cryptographically secure API key.

    Format: {prefix}_{32-char hex token}
    Example: cbm_prod_a3f8b2c1...

    Usage:
        new_key = generate_api_key("cbm_prod")
        # Add to .env: API_KEYS=existing_key,new_key
    """
    token = secrets.token_hex(16)
    return f"{prefix}_{token}"


def hash_key(api_key: str) -> str:
    """Return a salted SHA-256 hash of an API key (for safe storage)."""
    salt = os.getenv("KEY_HASH_SALT", "cbm_default_salt")
    return hashlib.sha256(f"{salt}{api_key}".encode()).hexdigest()


# ── CLI utility ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CBM API Key Manager")
    sub = parser.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate a new API key")
    gen.add_argument("--prefix", default="cbm", help="Key prefix (default: cbm)")
    gen.add_argument("--count",  type=int, default=1, help="Number of keys to generate")

    sub.add_parser("list", help="List configured API keys (masked)")

    args = parser.parse_args()

    if args.cmd == "generate":
        print(f"\nGenerated {args.count} API key(s):\n")
        for _ in range(args.count):
            key = generate_api_key(args.prefix)
            print(f"  {key}")
        print(f"\nAdd to your .env file:")
        print(f"  API_KEYS=key1,key2,...\n")

    elif args.cmd == "list":
        keys = _get_keys()
        print(f"\nConfigured API keys ({len(keys)}):")
        for k in sorted(keys):
            masked = k[:6] + "***" + k[-3:] if len(k) > 9 else "***"
            print(f"  {masked}")

    else:
        parser.print_help()
