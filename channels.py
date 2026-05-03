"""
channels.py — Async messaging channels for the AI harness.

Enables the harness to receive messages from external services
(Telegram, Slack, etc.) and process them through the agent loop.

Each channel polls for messages, feeds them to the agent, and
sends responses back. Multiple channels can run simultaneously
alongside the terminal REPL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger("agent_harness.channels")


@dataclass
class IncomingMessage:
    channel: str
    chat_id: str
    text: str
    user_name: str = ""
    raw: dict = field(default_factory=dict)


class MessageChannel(ABC):
    name: str = "base"

    @abstractmethod
    def poll(self) -> list[IncomingMessage]:
        pass

    @abstractmethod
    def send(self, chat_id: str, text: str) -> None:
        pass

    def send_status(self, chat_id: str, status: str) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class TelegramChannel(MessageChannel):
    name = "telegram"

    def __init__(self, token: str, allowed_chats: list[str] = None):
        self._token = token
        self._allowed = set(str(c) for c in (allowed_chats or []))
        self._base = f"https://api.telegram.org/bot{token}"
        self._offset = 0
        self._status_msg_ids: dict[str, int] = {}

    def start(self) -> None:
        try:
            r = requests.get(f"{self._base}/getMe", timeout=10)
            data = r.json()
            if data.get("ok"):
                bot_name = data["result"].get("username", "unknown")
                logger.info("Telegram connected: @%s", bot_name)
            else:
                logger.error("Telegram auth failed: %s", data)
        except Exception as exc:
            logger.error("Telegram connection failed: %s", exc)

    def poll(self) -> list[IncomingMessage]:
        try:
            r = requests.get(
                f"{self._base}/getUpdates",
                params={"offset": self._offset, "timeout": 5},
                timeout=10,
            )
            data = r.json()
            if not data.get("ok"):
                return []

            messages = []
            for update in data.get("result", []):
                self._offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "")
                if not text:
                    continue
                chat_id = str(msg.get("chat", {}).get("id", ""))
                user = msg.get("from", {})
                user_name = user.get("first_name", user.get("username", ""))

                if self._allowed and chat_id not in self._allowed:
                    logger.warning("Telegram: ignored message from unauthorized chat %s", chat_id)
                    continue

                messages.append(IncomingMessage(
                    channel="telegram",
                    chat_id=chat_id,
                    text=text,
                    user_name=user_name,
                    raw=msg,
                ))
            return messages
        except requests.Timeout:
            return []
        except Exception as exc:
            logger.warning("Telegram poll error: %s", exc)
            return []

    def send(self, chat_id: str, text: str) -> None:
        if len(text) > 4000:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        else:
            chunks = [text]

        for chunk in chunks:
            try:
                requests.post(
                    f"{self._base}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk},
                    timeout=10,
                )
            except Exception as exc:
                logger.warning("Telegram send error: %s", exc)

        self._status_msg_ids.pop(chat_id, None)

    def send_status(self, chat_id: str, status: str) -> None:
        msg_id = self._status_msg_ids.get(chat_id)
        try:
            if msg_id:
                requests.post(
                    f"{self._base}/editMessageText",
                    json={"chat_id": chat_id, "message_id": msg_id, "text": status},
                    timeout=5,
                )
            else:
                r = requests.post(
                    f"{self._base}/sendMessage",
                    json={"chat_id": chat_id, "text": status},
                    timeout=5,
                )
                data = r.json()
                if data.get("ok"):
                    self._status_msg_ids[chat_id] = data["result"]["message_id"]
        except Exception:
            pass

    def stop(self) -> None:
        logger.info("Telegram channel stopped")


DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


class ChannelRouter:
    def __init__(
        self,
        channels: list[MessageChannel],
        agent_factory: Callable,
        poll_interval: float = 1.0,
    ):
        self._channels = {c.name: c for c in channels}
        self._agent_factory = agent_factory
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        for ch in self._channels.values():
            ch.start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="channel-router")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for ch in self._channels.values():
            ch.stop()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for ch in self._channels.values():
                try:
                    messages = ch.poll()
                    for msg in messages:
                        self._handle_message(ch, msg)
                except Exception as exc:
                    logger.error("Channel %s error: %s", ch.name, exc)
            self._stop_event.wait(timeout=self._poll_interval)

    def _handle_message(self, channel: MessageChannel, msg: IncomingMessage) -> None:
        start = time.time()

        # Box header
        print(f"\n  {DIM}┌ {CYAN}{channel.name}{RESET}{DIM} · {msg.user_name}{RESET}")
        print(f"  {DIM}│{RESET} {msg.text}")

        channel.send_status(msg.chat_id, "⏳ Working...")

        try:
            agent_loop, history = self._agent_factory()
            history.clear()

            answer = agent_loop.run(msg.text)

            channel.send_status(msg.chat_id, "✅ Done")
            time.sleep(0.3)

            channel.send(msg.chat_id, answer)

            elapsed = int(time.time() - start)
            preview = answer[:80].replace("\n", " ")
            if len(answer) > 80:
                preview += "..."
            print(f"  {DIM}│{RESET} {GREEN}→{RESET} {DIM}{preview}{RESET}")
            print(f"  {DIM}└ {elapsed}s · {len(answer)} chars{RESET}")

        except Exception as exc:
            error_msg = f"Error: {exc}"
            channel.send(msg.chat_id, error_msg)
            elapsed = int(time.time() - start)
            print(f"  {DIM}│{RESET} {MAGENTA}✗{RESET} {DIM}{exc}{RESET}")
            print(f"  {DIM}└ {elapsed}s · error{RESET}")


def load_channels(config_path: str) -> list[MessageChannel]:
    with open(config_path) as f:
        config = json.load(f)

    channels = []
    for name, cfg in config.get("channels", {}).items():
        if name == "telegram":
            token = cfg.get("token", "")
            if token.startswith("$"):
                token = os.environ.get(token.lstrip("$"), "")
            if not token:
                logger.warning("Telegram: no token provided, skipping")
                continue
            allowed = cfg.get("allowed_chats", [])
            channels.append(TelegramChannel(token=token, allowed_chats=allowed))
        else:
            logger.warning("Unknown channel type: %s", name)

    return channels
