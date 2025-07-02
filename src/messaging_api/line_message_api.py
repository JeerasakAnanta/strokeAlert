import os
import requests

# Rag System
from src.rag_system.sql_agent import sql_agent
from src.rag_system.sql_agent import assistant_agent


# LINE Messaging API
import json
import hmac
import hashlib
import requests
import base64
from starlette.responses import JSONResponse
import logging

from dotenv import load_dotenv

load_dotenv()


# Constants for LINE API
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")


def verify_signature(request_body: str, signature: str) -> bool:
    """
    Verifies that the incoming request comes from LINE by validating the HMAC SHA256 signature.
    """
    hash_mac = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"),
        request_body.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    expected_signature = base64.b64encode(hash_mac).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


async def get_sqlchat_response(user_message: str) -> str:
    """
    Sends user messages to the ChatGPT model and retrieves a response.
    """
    try:
        sql_result = sql_agent(user_message)
        output_assistant_agent = assistant_agent(sql_result)
        return output_assistant_agent
    except Exception as e:
        return "I'm sorry, I couldn't process your request."


async def reply_to_user(reply_token: str, text: str):
    """
    Sends a reply to the user through the LINE Messaging API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    response = requests.post(
        "https://api.line.me/v2/bot/message/reply", json=payload, headers=headers
    )


async def line_chatbot(user_message: str) -> str:
    """
    Handles incoming webhook requests from the LINE Messaging API.
    """
    signature = request.headers.get("X-Line-Signature")
    request_body = await request.body()

    # Verify the request signature
    if not verify_signature(request_body.decode("utf-8"), signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    data = await request.json()
    logging.info(f"Received LINE webhook: {json.dumps(data, indent=2)}")

    # Process each event
    for event in data.get("events", []):
        if event.get("type") == "message":
            user_id = event["source"]["userId"]
            reply_token = event["replyToken"]
            user_message = event["message"]["text"]

            logging.info(f"User ({user_id}) sent: {user_message}")

            # Get ChatGPT response and reply to the user
            chatgpt_response = await get_sqlchat_response(user_message)
            await reply_to_user(reply_token, chatgpt_response)

    return JSONResponse(
        content={"status": "success", "message": "Webhook received"}, status_code=200
    )
