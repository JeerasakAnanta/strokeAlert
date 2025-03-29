from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
import hmac
import hashlib
import os
from starlette.responses import JSONResponse

# Environment Variables (Set these before running)
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_ACCESS_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def verify_signature(request_body: str, signature: str) -> bool:
    """Verify that the request comes from LINE using HMAC SHA256."""
    hash_mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), request_body.encode("utf-8"), hashlib.sha256).digest()
    expected_signature = hashlib.base64.b64encode(hash_mac).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)

@app.post("/webhook")
async def line_webhook(request: Request):
    """Handle incoming webhook requests from LINE Messaging API."""
    signature = request.headers.get("X-Line-Signature")
    request_body = await request.body()

    # Verify the request signature
    if not verify_signature(request_body.decode("utf-8"), signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    data = await request.json()
    logging.info(f"Received LINE webhook: {json.dumps(data, indent=2)}")

    # Process each event
    for event in data.get("events", []):
        event_type = event.get("type")

        if event_type == "message":
            message_type = event["message"]["type"]
            user_id = event["source"]["userId"]
            reply_token = event["replyToken"]

            if message_type == "text":
                user_message = event["message"]["text"]
                logging.info(f"User ({user_id}) sent: {user_message}")

                # TODO: Reply to the user (implement reply function)
                await reply_to_user(reply_token, f"You said: {user_message}")

    return JSONResponse(content={"status": "success", "message": "Webhook received"}, status_code=200)

async def reply_to_user(reply_token: str, text: str):
    """Reply to user using LINE Messaging API (implement this function)."""
    import requests

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }

    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}]
    }

    response = requests.post("https://api.line.me/v2/bot/message/reply", json=payload, headers=headers)

    if response.status_code == 200:
        logging.info("Reply sent successfully!")
    else:
        logging.error(f"Failed to send reply: {response.text}")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
