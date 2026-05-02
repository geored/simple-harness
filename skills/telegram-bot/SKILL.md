---
description: Guides the agent on how to connect to Telegram, send messages, receive
  updates, and interact with users or groups via the Telegram Bot API. Use when asked
  to send Telegram messages, build a Telegram bot, poll for updates, or automate Telegram
  communication.
metadata:
  author: ai-generated
  version: '1.0'
name: telegram-bot
---

# Telegram Bot Skill

## Overview
This skill enables the agent to interact with Telegram using the **Telegram Bot API** (no third-party libraries required — pure HTTP). It covers setup, sending messages, receiving updates, and best practices.

---

## Step 1: Prerequisites — Create a Bot Token

Before any API call, the user must have a **Bot Token**:
1. Open Telegram and message **@BotFather**
2. Send `/newbot` and follow the prompts (name + username)
3. BotFather returns a token like: `123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
4. Store it as an environment variable: `TELEGRAM_BOT_TOKEN`

> ⚠️ **Never hardcode the token in code.** Always read from env: `os.environ["TELEGRAM_BOT_TOKEN"]`

---

## Step 2: Base API URL

All Telegram Bot API calls follow this pattern:
```
https://api.telegram.org/bot<TOKEN>/<METHOD>
```

Example:
```
https://api.telegram.org/bot123456:ABC-DEF/sendMessage
```

---

## Step 3: Verify the Bot is Alive — `getMe`

Use `http_fetch` or instruct the user to run:
```
GET https://api.telegram.org/bot<TOKEN>/getMe
```
Returns bot info (username, id, etc.). Confirms the token is valid.

---

## Step 4: Get the Chat ID

To send a message, you need a **chat_id**. To find it:
1. Send any message to your bot in Telegram
2. Call `getUpdates`:
```
GET https://api.telegram.org/bot<TOKEN>/getUpdates
```
3. Parse the response: `result[0].message.chat.id`

> For groups: add the bot to the group, send a message, then call `getUpdates`.

---

## Step 5: Send a Message — `sendMessage`

```
POST https://api.telegram.org/bot<TOKEN>/sendMessage
Content-Type: application/json

{
  "chat_id": "<CHAT_ID>",
  "text": "Hello from the bot!",
  "parse_mode": "Markdown"   // optional: "Markdown" or "HTML"
}
```

### Key Parameters:
| Parameter | Type | Description |
|---|---|---|
| `chat_id` | int/str | Target chat or username |
| `text` | str | Message content (max 4096 chars) |
| `parse_mode` | str | `Markdown`, `MarkdownV2`, or `HTML` |
| `disable_notification` | bool | Silent message |
| `reply_to_message_id` | int | Reply to a specific message |

---

## Step 6: Poll for Incoming Messages — `getUpdates`

```
GET https://api.telegram.org/bot<TOKEN>/getUpdates?offset=<OFFSET>&timeout=30
```

- `offset`: Set to `last_update_id + 1` to avoid re-processing
- `timeout`: Long-polling duration in seconds (0 = short poll)

### Polling Loop Pattern (Python pseudocode):
```python
import os, requests, time

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
BASE = f"https://api.telegram.org/bot{TOKEN}"
offset = 0

while True:
    resp = requests.get(f"{BASE}/getUpdates", params={"offset": offset, "timeout": 30})
    updates = resp.json().get("result", [])
    for update in updates:
        offset = update["update_id"] + 1
        msg = update.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text", "")
        # Handle the message here
        requests.post(f"{BASE}/sendMessage", json={"chat_id": chat_id, "text": f"Echo: {text}"})
    time.sleep(1)
```

---

## Step 7: Additional Useful Methods

| Method | Purpose |
|---|---|
| `sendMessage` | Send a text message |
| `sendPhoto` | Send an image |
| `sendDocument` | Send a file |
| `sendPoll` | Create a poll |
| `editMessageText` | Edit a sent message |
| `deleteMessage` | Delete a message |
| `getChat` | Get info about a chat/group |
| `getChatMembers` | List group members |
| `setWebhook` | Set a webhook URL instead of polling |

---

## Step 8: Webhook (Alternative to Polling)

For production, use **webhooks** instead of polling:
```
POST https://api.telegram.org/bot<TOKEN>/setWebhook
{
  "url": "https://yourdomain.com/webhook"
}
```
Telegram will POST updates to your URL in real time. Requires HTTPS.

---

## Security Best Practices
- ✅ Store token in environment variables only
- ✅ Validate `chat_id` before sending (whitelist known IDs)
- ✅ Use HTTPS for webhooks
- ✅ Set a **secret token** on webhooks to verify requests come from Telegram
- ❌ Never log the full bot token
- ❌ Never commit `.env` files to version control

---

## Error Handling

Always check the `ok` field in API responses:
```json
{ "ok": false, "error_code": 400, "description": "Bad Request: chat not found" }
```

Common errors:
| Code | Meaning |
|---|---|
| 400 | Bad request (invalid params) |
| 401 | Invalid token |
| 403 | Bot was blocked by user |
| 429 | Rate limited — respect `retry_after` |

---

## Agent Action Checklist
When a user asks to connect to Telegram:
1. ✅ Ask for or confirm `TELEGRAM_BOT_TOKEN` is set in env
2. ✅ Fetch `getMe` to verify the bot is alive
3. ✅ Fetch `getUpdates` to discover `chat_id` if not known
4. ✅ Use `sendMessage` to send the desired content
5. ✅ If building a bot loop — implement offset-based polling or webhook
6. ✅ Always handle errors from the API response

