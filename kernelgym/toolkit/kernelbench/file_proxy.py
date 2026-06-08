"""
File-based Proxy for KernelGym — replaces HTTP when nodes share a filesystem
but have no network connectivity.

Protocol:
  REQUEST:  {shared_dir}/requests/{task_id}.json   (Client -> Server)
  RESPONSE: {shared_dir}/responses/{task_id}.json   (Server -> Client)
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


# ---- helpers -----------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and other non-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def _atomic_write(path: str, data: Any) -> None:
    """Write JSON to *path* atomically via a .tmp rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_to_json_safe(data), fh)
    os.rename(tmp, path)


# ---- client (RL training node) -----------------------------------------------

def send_eval_via_file(
    shared_dir: str,
    payload: Dict[str, Any],
    poll_interval: float = 1.0,
    client_timeout: float = 600.0,
) -> Dict[str, Any]:
    """Write request file and block until the response appears.

    Returns a dict matching the HTTP eval response format.
    """
    task_id = payload.get("task_id", "")
    requests_dir = os.path.join(shared_dir, "requests")
    responses_dir = os.path.join(shared_dir, "responses")
    os.makedirs(requests_dir, exist_ok=True)
    os.makedirs(responses_dir, exist_ok=True)

    request_path = os.path.join(requests_dir, f"{task_id}.json")
    response_path = os.path.join(responses_dir, f"{task_id}.json")

    _atomic_write(request_path, payload)
    logger.info("[FileProxy] request: %s", request_path)

    deadline = time.time() + client_timeout
    while time.time() < deadline:
        if os.path.exists(response_path):
            try:
                with open(response_path) as fh:
                    result = json.load(fh)
                logger.info("[FileProxy] response: %s", response_path)
                os.remove(response_path)
                os.remove(request_path)
                return result
            except Exception:
                logger.warning("[FileProxy] read error, retrying")
        time.sleep(poll_interval)

    # Timeout — clean up so the server doesn't process a stale request.
    if os.path.exists(request_path):
        try:
            os.remove(request_path)
        except OSError:
            pass
    return {"status": "timeout", "error_message": f"FileProxy timeout after {client_timeout:.0f}s"}


# ---- server-side proxy (eval node) --------------------------------------------

def run_file_proxy(
    shared_dir: str,
    server_url: str = "http://localhost:8001",
    poll_interval: float = 0.5,
) -> None:
    """Poll the shared requests directory and forward each request to the local
    HTTP API server, then write the response back to the shared directory.

    The existing HTTP API server (Redis, workers, GPU workers) runs
    independently — this proxy is just a thin bridge between the shared
    filesystem and the local HTTP endpoint.
    """
    import httpx

    requests_dir = os.path.join(shared_dir, "requests")
    responses_dir = os.path.join(shared_dir, "responses")
    os.makedirs(requests_dir, exist_ok=True)
    os.makedirs(responses_dir, exist_ok=True)

    client = httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=10.0, pool=5.0),
        headers={"Content-Type": "application/json"},
    )

    logger.info("[FileProxy] started  server=%s  dir=%s", server_url, shared_dir)

    while True:
        try:
            entries = sorted(
                (os.path.join(requests_dir, f) for f in os.listdir(requests_dir)
                 if f.endswith(".json") and not f.endswith(".tmp")),
                key=os.path.getctime,
            )
            for request_path in entries:
                processing_path = request_path + ".processing"
                try:
                    os.rename(request_path, processing_path)
                except OSError:
                    continue

                task_id = "unknown"
                try:
                    with open(processing_path) as fh:
                        payload = json.load(fh)
                    task_id = payload.get("task_id", "unknown")
                    logger.info("[FileProxy] forwarding  task_id=%s", task_id)

                    result = _post_with_retry(client, server_url, payload)

                except Exception:
                    logger.exception("[FileProxy] error  task_id=%s", task_id)
                    result = {"task_id": task_id, "status": "failed",
                              "error_message": traceback.format_exc()}
                finally:
                    try:
                        os.remove(processing_path)
                    except OSError:
                        pass

                response_path = os.path.join(responses_dir, f"{result.get('task_id', 'unknown')}.json")
                _atomic_write(response_path, result)

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("[FileProxy] interrupted, shutting down")
            break
        except Exception:
            logger.exception("[FileProxy] loop error")
            time.sleep(poll_interval)
def _post_with_retry(
    client: Any, server_url: str, payload: Dict[str, Any],
    max_retries: int = 5, backoff: float = 2.0,
) -> Dict[str, Any]:
    """POST to the API with retry on connection errors (server not ready yet)."""
    import httpx

    task_id = payload.get("task_id", "unknown")
    for attempt in range(max_retries + 1):
        try:
            resp = client.post(f"{server_url}/evaluate", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            if attempt < max_retries:
                wait = backoff ** attempt
                logger.warning("[FileProxy] connection refused, retry %d/%d in %.1fs  task_id=%s",
                               attempt + 1, max_retries, wait, task_id)
                time.sleep(wait)
            else:
                raise
        # Other errors (timeout, HTTP 5xx etc.) are not retried — the RL client
        # will see the failure and its own retry logic will handle it.

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    shared_dir = os.environ.get("KERNELGYM_SHARED_DIR", "/tmp/kernelgym_shared")
    server_url = os.environ.get("KERNELGYM_SERVER_URL", "http://localhost:8001")
    poll = float(os.environ.get("KERNELGYM_POLL_INTERVAL", "0.5"))
    run_file_proxy(shared_dir, server_url, poll)
