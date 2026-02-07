"""In-memory file store tool for read/write of small artifacts."""

from __future__ import annotations

from typing import Any

from promptbench.tools.base import Tool


class FileStoreTool(Tool):
    """In-memory key-value store for reading and writing small artifacts.

    Supports read, write, list, and delete operations.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "file_store"

    @property
    def description(self) -> str:
        return (
            "Read, write, list, or delete small text artifacts in a key-value store. "
            "Actions: 'read' (key), 'write' (key, value), 'list' (), 'delete' (key)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list", "delete"],
                    "description": "Operation to perform",
                },
                "key": {
                    "type": "string",
                    "description": "Key for the artifact (required for read/write/delete)",
                },
                "value": {
                    "type": "string",
                    "description": "Value to store (required for write)",
                },
            },
            "required": ["action"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute file store operation."""
        action = arguments.get("action", "")
        key = arguments.get("key", "")
        value = arguments.get("value", "")

        if action == "write":
            if not key:
                raise ValueError("Missing 'key' for write operation")
            if len(value) > 10000:
                raise ValueError("Value too large (max 10000 characters)")
            self._store[key] = value
            return f"Written to '{key}' ({len(value)} chars)"

        elif action == "read":
            if not key:
                raise ValueError("Missing 'key' for read operation")
            if key not in self._store:
                raise ValueError(f"Key not found: '{key}'")
            return self._store[key]

        elif action == "list":
            if not self._store:
                return "Store is empty."
            keys = sorted(self._store.keys())
            return "Keys: " + ", ".join(keys)

        elif action == "delete":
            if not key:
                raise ValueError("Missing 'key' for delete operation")
            if key not in self._store:
                raise ValueError(f"Key not found: '{key}'")
            del self._store[key]
            return f"Deleted '{key}'"

        else:
            raise ValueError(f"Unknown action: '{action}'. Use: read, write, list, delete")

    def reset(self) -> None:
        """Clear all stored artifacts."""
        self._store.clear()
