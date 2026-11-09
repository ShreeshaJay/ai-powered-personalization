"""
Service-to-service authentication on Cloud Run (SERVICE_AUTH=gcp_id_token).

The stage services are deployed without public access, so the orchestrator
attaches a Google-signed ID token whose audience is the callee's URL. The
token comes from the instance metadata server of the orchestrator's runtime
service account; no key file or secret is stored anywhere. Tokens are cached
until shortly before they expire. Locally, SERVICE_AUTH defaults to none.
"""

import asyncio
import time

import httpx

METADATA_IDENTITY_URL = ("http://metadata.google.internal/computeMetadata/v1/"
                         "instance/service-accounts/default/identity")
TOKEN_LIFETIME_S = 50 * 60


class GcpIdTokenAuth(httpx.Auth):
    def __init__(self, audience: str):
        self.audience = audience.rstrip("/")
        self._token = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def _token_value(self) -> str:
        async with self._lock:
            if self._token is None or time.monotonic() >= self._expires_at:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(METADATA_IDENTITY_URL, params={"audience": self.audience},
                                                headers={"Metadata-Flavor": "Google"})
                    response.raise_for_status()
                    self._token = response.text.strip()
                    self._expires_at = time.monotonic() + TOKEN_LIFETIME_S
            return self._token

    async def async_auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {await self._token_value()}"
        yield request
