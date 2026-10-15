"""HTTP client for chat completions, with retries for transient failures.

Shared by the LM Studio and PostQode model backends. A single blip previously
discarded an entire trajectory: one arm losing a step meant the whole cell was
abandoned and its trace never written. Three of nine arms were lost this way
in one run, all failing within the same second while two requests ran
concurrently.
"""

import json
import logging
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger("completion")

MAX_COMPLETION_ATTEMPTS = 5
# A local server that is unreachable (the Turing vLLM job being replaced at its
# walltime, the SSH tunnel re-establishing) is waited out for this long before
# the cell is failed. Only connection-level failures on a local endpoint count;
# cloud requests keep the plain attempt budget.
LOCAL_OUTAGE_WAIT_S = float(os.environ.get("LOCAL_OUTAGE_WAIT_S", 900))
# A TLS fault (a corrupted record, a reset mid-handshake) comes in bursts on a
# flaky path and says nothing about the request; it is waited out this long
# for any endpoint before the cell is failed. A DNS failure counts too.
TRANSPORT_FAULT_WAIT_S = float(os.environ.get("TRANSPORT_FAULT_WAIT_S", 1800))


def post_completion(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    max_attempts: int = MAX_COMPLETION_ATTEMPTS,
) -> Dict[str, Any]:
    """POST a completion request, retrying transient failures.

    Retries cover network faults, 5xx and 429, and also 400 from a local
    server, which LM Studio returns for transient states such as a model being
    swapped or reloaded under concurrent load. A 400 from the cloud gateway is
    treated as a genuine client error and raised immediately, since retrying a
    malformed or oversized request only wastes the budget; its body is logged
    so the cause is visible.
    """
    is_local = "localhost" in url or "127.0.0.1" in url
    delay = 2.0
    last_exc: Optional[Exception] = None
    outage_since: Optional[float] = None

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            if exc.code == 400 and ("context length" in body.lower() or "max_model_len" in body.lower()):
                logger.warning("Context length exceeded: %s. Terminating this task.", body)
                raise RuntimeError(f"Context length exceeded: {body}")
            retryable = exc.code >= 500 or exc.code == 429 or (exc.code == 400 and is_local)
            if not retryable:
                if exc.code == 401:
                    # The caller may recover (another process refreshed the
                    # token); a real auth failure surfaces when its retry fails.
                    logger.warning("Completion unauthorized: %s", body[:120])
                else:
                    logger.error("Completion failed permanently (HTTP %s): %s", exc.code, body)
                raise
            last_exc = exc
            logger.warning("HTTP %s on attempt %d/%d: %s", exc.code, attempt, max_attempts, body)
        except Exception as exc:
            last_exc = exc
            unreachable = isinstance(exc, (urllib.error.URLError, ConnectionError)) or \
                "Remote end closed" in str(exc)
            reason = getattr(exc, "reason", None)
            # A failed DNS lookup means this machine lost its network, not that
            # the request is bad: waited out like a TLS fault, for any endpoint.
            transport = isinstance(exc, ssl.SSLError) or isinstance(reason, (ssl.SSLError, socket.gaierror))
            if (is_local and unreachable) or transport:
                outage_since = outage_since or time.monotonic()
                waited = time.monotonic() - outage_since
                if waited < (LOCAL_OUTAGE_WAIT_S if is_local and unreachable else TRANSPORT_FAULT_WAIT_S):
                    logger.warning("Local server unreachable (%.0fs so far): %r; waiting", waited, exc)
                    attempt -= 1  # an outage does not use up the attempt budget
                    time.sleep(15)
                    continue
            logger.warning("Completion error on attempt %d/%d: %r", attempt, max_attempts, exc)

        if attempt < max_attempts:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    raise RuntimeError(f"Completion failed after {max_attempts} attempts: {last_exc!r}")
