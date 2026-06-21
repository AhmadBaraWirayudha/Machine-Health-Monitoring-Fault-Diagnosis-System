"""
API Middleware
==============
Request logging, timing, and error handling middleware for FastAPI.

Middleware registered:
  RequestLoggingMiddleware  — logs every request with method, path,
                              status code, duration, and client IP
  ErrorHandlingMiddleware   — catches unhandled exceptions and returns
                              structured JSON error responses

Register in src/api/main.py:
    from src.api.middleware import RequestLoggingMiddleware, add_middleware
    add_middleware(app)
"""

import time
import uuid
import logging
import traceback
from typing import Callable

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp
except ImportError:
    raise ImportError("Install FastAPI: pip install fastapi uvicorn")

log = logging.getLogger("cbm.api")

# ── Request ID header name ────────────────────────────────────────────────────
REQUEST_ID_HEADER = "X-Request-ID"

# ── Paths excluded from verbose logging ───────────────────────────────────────
QUIET_PATHS = {"/health", "/", "/favicon.ico", "/nginx-health"}


# ── Request Logging Middleware ────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs each HTTP request with:
      - Unique request ID (X-Request-ID header)
      - HTTP method + path + query string
      - Client IP
      - Response status code
      - Processing duration (ms)
      - Optional: API key prefix (if auth enabled)

    Log format:
      2025-06-20 14:22:31 [INFO] GET /predict 200 5.4ms [req=abc123] [ip=127.0.0.1]
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4())[:8])
        start_time = time.perf_counter()

        # Attach request ID to state for downstream use
        request.state.request_id = request_id

        # Log incoming request (skip noisy health checks)
        path = request.url.path
        if path not in QUIET_PATHS:
            qs = f"?{request.url.query}" if request.url.query else ""
            log.info(
                f"→ {request.method} {path}{qs} "
                f"[req={request_id}] [ip={_get_client_ip(request)}]"
            )

        # Process request
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.error(
                f"✗ {request.method} {path} 500 {duration_ms:.1f}ms "
                f"[req={request_id}] UNHANDLED: {exc}"
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log response
        status = response.status_code
        level  = logging.WARNING if status >= 400 else logging.INFO
        if path not in QUIET_PATHS:
            log.log(
                level,
                f"← {request.method} {path} {status} {duration_ms:.1f}ms "
                f"[req={request_id}]"
            )

        # Add response headers
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time"]  = f"{duration_ms:.2f}ms"
        return response


# ── Error Handling Middleware ─────────────────────────────────────────────────

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Catches any unhandled exception and returns a consistent JSON error body
    instead of a raw 500 traceback.

    Production mode: hides internal details.
    Debug mode:      includes traceback in response (set DEBUG=true in .env).
    """

    def __init__(self, app: ASGIApp, debug: bool = False):
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = getattr(request.state, "request_id", "unknown")
            log.error(
                f"[req={request_id}] Unhandled exception on "
                f"{request.method} {request.url.path}:\n"
                f"{traceback.format_exc()}"
            )
            body = {
                "error":      "Internal server error",
                "request_id": request_id,
                "path":       request.url.path,
            }
            if self.debug:
                body["detail"] = str(exc)
                body["traceback"] = traceback.format_exc().splitlines()
            return JSONResponse(status_code=500, content=body)


# ── CORS + Security Headers Middleware ───────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to every response:
      X-Content-Type-Options, X-Frame-Options, Referrer-Policy,
      Content-Security-Policy (basic), Cache-Control for API endpoints.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]         = "DENY"
        response.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"]         = "1; mode=block"

        # Prevent caching of API responses
        if request.url.path.startswith("/predict") or \
           request.url.path.startswith("/fleet") or \
           request.url.path.startswith("/assets"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response


# ── Metrics Middleware ────────────────────────────────────────────────────────

class SimpleMetricsMiddleware(BaseHTTPMiddleware):
    """
    In-memory request counter and latency tracker.
    Exposed via GET /metrics (add a route manually or use /health).

    For production: replace with Prometheus middleware (prometheus-fastapi-instrumentator).
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.total_requests:  int   = 0
        self.total_errors:    int   = 0
        self.total_latency_ms: float = 0.0
        self.status_counts:   dict  = {}
        self.path_counts:     dict  = {}

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_ms / self.total_requests
                if self.total_requests > 0 else 0.0)

    def snapshot(self) -> dict:
        return {
            "total_requests":  self.total_requests,
            "total_errors":    self.total_errors,
            "avg_latency_ms":  round(self.avg_latency_ms, 2),
            "status_counts":   dict(self.status_counts),
            "top_paths":       dict(
                sorted(self.path_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        self.total_requests += 1
        self.total_latency_ms += duration_ms
        status = str(response.status_code)
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        if response.status_code >= 500:
            self.total_errors += 1

        path = request.url.path
        self.path_counts[path] = self.path_counts.get(path, 0) + 1
        return response


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """Extract real client IP, handling X-Forwarded-For (behind proxy)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Registration helper ───────────────────────────────────────────────────────

def add_middleware(app: FastAPI, debug: bool = False) -> SimpleMetricsMiddleware:
    """
    Register all CBM middleware on the FastAPI app.

    Call this in src/api/main.py after creating the app:
        from src.api.middleware import add_middleware
        metrics = add_middleware(app)

    Returns the SimpleMetricsMiddleware instance for use in a /metrics route.
    """
    import os
    debug = debug or os.getenv("DEBUG", "false").lower() == "true"

    metrics_mw = SimpleMetricsMiddleware(app)

    # Add in reverse order (last added = outermost wrapper)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ErrorHandlingMiddleware, debug=debug)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(type(metrics_mw).__mro__[0], app=app)

    log.info("[middleware] Registered: Logging, ErrorHandling, SecurityHeaders")
    return metrics_mw
