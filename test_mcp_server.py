#!/usr/bin/env python3
"""
Minimal MCP server over STDIO for testing.
Exposes three tools: echo, uppercase, word_count.
"""
import json
import sys


TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the input text unchanged. Input: text (string).",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
            "required": ["text"],
        },
    },
    {
        "name": "uppercase",
        "description": "Convert text to uppercase. Input: text (string).",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to convert"}},
            "required": ["text"],
        },
    },
    {
        "name": "word_count",
        "description": "Count the number of words in text. Input: text (string).",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to count words in"}},
            "required": ["text"],
        },
    },
]


def handle_request(req):
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-server", "version": "1.0"},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        text = args.get("text", "")

        if name == "echo":
            result = text
        elif name == "uppercase":
            result = text.upper()
        elif name == "word_count":
            result = str(len(text.split()))
        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }

        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": result}]},
        }

    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
