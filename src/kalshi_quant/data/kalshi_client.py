"""Kalshi REST + WebSocket API client.

Auth: Kalshi API v2 uses an RSA key-pair scheme, not a static bearer
token. Every request is signed:

    message   = f"{timestamp_ms}{METHOD}{path}"   # path only, no query string
    signature = base64(RSA_PSS_SHA256_sign(private_key, message))

sent as headers `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-SIGNATURE`,
`KALSHI-ACCESS-TIMESTAMP`. Credentials are read from environment
variables (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`) -- never
hardcoded, never logged.

NOTE ON API STABILITY: this scheme, the base URLs, and the endpoint
paths below were confirmed against Kalshi's official
`kalshi-starter-code-python` reference client. Kalshi has changed its
production hostname before (the prod API currently lives at
api.elections.kalshi.com despite covering all categories, not just
elections) and could again. If requests start failing with auth errors
after working previously, re-verify this module against Kalshi's
current docs/starter code before assuming the strategy is broken.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import requests
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

DEMO_BASE_URL = "https://demo-api.kalshi.co"
PROD_BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"

_MIN_REQUEST_INTERVAL_SECONDS = 0.10  # Kalshi's documented per-key rate limit floor


class KalshiAPIError(RuntimeError):
    """Raised for non-2xx responses that are not worth retrying."""

    def __init__(self, status_code: int, body: str, method: str, path: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Kalshi API error {status_code} on {method} {path}: {body}")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.ConnectionError | requests.exceptions.Timeout):
        return True
    if isinstance(exc, KalshiAPIError):
        # Retry server errors and explicit rate-limit responses; never retry
        # 4xx client errors (bad request, auth failure, not found) since a
        # retry cannot fix a malformed request and could duplicate an order.
        return exc.status_code == 429 or exc.status_code >= 500
    return False


@dataclass(frozen=True)
class KalshiCredentials:
    key_id: str
    private_key: rsa.RSAPrivateKey

    @classmethod
    def from_env(cls) -> "KalshiCredentials":
        key_id = os.environ.get("KALSHI_API_KEY_ID")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not key_path:
            raise RuntimeError(
                "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set "
                "(see .env.example). Refusing to start with missing credentials."
            )
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise RuntimeError(f"Key at {key_path} is not an RSA private key")
        return cls(key_id=key_id, private_key=private_key)


class KalshiClient:
    """Thin, explicit wrapper around the Kalshi v2 REST API.

    Every public method returns plain dicts/lists straight from the JSON
    response -- mapping into the dataclasses in data/models.py (with an
    `as_of`/`fetched_at` timestamp stamped at receipt time) is the
    caller's job, so this class has exactly one responsibility:
    authenticated, retried, rate-limited HTTP.
    """

    def __init__(
        self,
        credentials: KalshiCredentials | None = None,
        environment: str | None = None,
        session: requests.Session | None = None,
    ):
        self._credentials = credentials or KalshiCredentials.from_env()
        env = environment or os.environ.get("KALSHI_ENVIRONMENT", "demo")
        if env not in ("demo", "prod"):
            raise ValueError(f"KALSHI_ENVIRONMENT must be 'demo' or 'prod', got {env!r}")
        self.environment = env
        self.base_url = DEMO_BASE_URL if env == "demo" else PROD_BASE_URL
        self._session = session or requests.Session()
        self._last_request_at = 0.0

    # -- low-level request plumbing -----------------------------------

    def _sign(self, method: str, path: str, timestamp_ms: int) -> str:
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self._credentials.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
        reraise=True,
    )
    def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                 json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        """`path` must NOT include query string -- the signature only covers
        the path, so query params go through `params` and are added by
        `requests` after signing."""
        self._rate_limit()
        timestamp_ms = int(time.time() * 1000)
        full_path = f"{API_PREFIX}{path}"
        signature = self._sign(method, full_path, timestamp_ms)
        headers = {
            "KALSHI-ACCESS-KEY": self._credentials.key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}{full_path}"
        self._last_request_at = time.monotonic()
        response = self._session.request(
            method, url, headers=headers, params=params, json=json_body, timeout=15
        )
        if response.status_code >= 400:
            logger.warning(
                "kalshi_api_error",
                status=response.status_code,
                method=method,
                path=path,
                body=response.text[:500],
            )
            raise KalshiAPIError(response.status_code, response.text, method, path)
        return response.json() if response.content else {}

    def _paginate(self, path: str, params: dict[str, Any], items_key: str) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            page = self._request("GET", path, params=page_params)
            yield from page.get(items_key, [])
            cursor = page.get("cursor") or None
            if not cursor:
                return

    # -- exchange metadata ----------------------------------------------

    def get_exchange_status(self) -> dict[str, Any]:
        return self._request("GET", "/exchange/status")

    def get_exchange_schedule(self) -> dict[str, Any]:
        return self._request("GET", "/exchange/schedule")

    # -- market metadata --------------------------------------------------

    def get_series_list(self, category: str | None = None) -> list[dict[str, Any]]:
        params = {"category": category} if category else {}
        return list(self._paginate("/series", params, "series"))

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        return self._request("GET", f"/series/{series_ticker}")

    def get_events(self, series_ticker: str | None = None, status: str | None = None,
                    limit: int = 200) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        return self._paginate("/events", params, "events")

    def get_markets(self, event_ticker: str | None = None, series_ticker: str | None = None,
                     status: str | None = None, limit: int = 200) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        return self._paginate("/markets", params, "markets")

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 50) -> dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_trades(self, ticker: str | None = None, min_ts: int | None = None,
                    max_ts: int | None = None, limit: int = 200) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        return self._paginate("/markets/trades", params, "trades")

    # -- portfolio (requires auth, used by execution layer) --------------

    def get_balance(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self) -> Iterator[dict[str, Any]]:
        return self._paginate("/portfolio/positions", {}, "market_positions")

    def get_orders(self, status: str | None = None) -> Iterator[dict[str, Any]]:
        params = {"status": status} if status else {}
        return self._paginate("/portfolio/orders", params, "orders")

    def create_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """`order` must match Kalshi's CreateOrderRequest schema (ticker,
        action, side, count, type, yes_price/no_price, client_order_id).
        Callers (execution layer) are responsible for building it -- this
        method does not add defaults, so nothing can be sent live by
        accident via an implicit fallback."""
        return self._request("POST", "/portfolio/orders", json_body=order)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_fills(self, ticker: str | None = None, order_id: str | None = None,
                   limit: int = 200) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        return self._paginate("/portfolio/fills", params, "fills")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
