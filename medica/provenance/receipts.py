from __future__ import annotations

import hashlib
import json


def create_receipt(payload: dict) -> dict:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return {
        "algorithm": "sha256",
        "digest": digest,
        "payload": payload,
    }
