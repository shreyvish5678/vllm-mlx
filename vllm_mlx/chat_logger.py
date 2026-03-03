"""Chat conversation logger.

Persists chat conversations to JSONL files in a logs/ directory.
Each conversation gets its own file, identified by a hash of the
initial messages. Subsequent requests that extend the same conversation
(same prefix of messages) append to the same file.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("vllm_mlx.chat_logger")

# Directory for log files, relative to the working directory
_LOGS_DIR: Path | None = None
# Map conversation fingerprint -> log file path
_conversation_files: dict[str, Path] = {}


def _ensure_logs_dir() -> Path:
    global _LOGS_DIR
    if _LOGS_DIR is None:
        _LOGS_DIR = Path("logs")
    _LOGS_DIR.mkdir(exist_ok=True)
    return _LOGS_DIR


def _extract_text(content) -> str:
    """Extract plain text from message content (string or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content) if content else ""


def _conversation_fingerprint(messages: list[dict]) -> str:
    """Create a stable fingerprint from the first system + first user message.

    This identifies a conversation so that follow-up requests (which include
    the full prior history plus new messages) map to the same log file.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ("system", "user"):
            parts.append(f"{role}:{_extract_text(msg.get('content', ''))}")
            if role == "user":
                break  # Stop after first user message
    fingerprint = "|".join(parts)
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _get_log_file(fingerprint: str) -> Path:
    """Get or create the log file for a conversation."""
    if fingerprint in _conversation_files:
        return _conversation_files[fingerprint]

    logs_dir = _ensure_logs_dir()

    # Check for existing files with this fingerprint
    for f in logs_dir.glob(f"*_{fingerprint}.jsonl"):
        _conversation_files[fingerprint] = f
        return f

    # Create new file with timestamp prefix for sorting
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"{timestamp}_{fingerprint}.jsonl"
    _conversation_files[fingerprint] = path
    return path


def log_chat(messages: list[dict], assistant_response: str, metadata: dict | None = None) -> None:
    """Log a chat exchange to the conversation's JSONL file.

    Args:
        messages: The full messages array from the request.
        assistant_response: The assistant's response text.
        metadata: Optional extra fields (model, tokens, latency, etc.).
    """
    try:
        fingerprint = _conversation_fingerprint(messages)
        log_file = _get_log_file(fingerprint)

        entry = {
            "timestamp": time.time(),
            "messages": messages,
            "assistant": assistant_response,
        }
        if metadata:
            entry["metadata"] = metadata

        with open(log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.debug(f"Logged chat to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log chat: {e}")
