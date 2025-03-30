from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
import hmac
import hashlib
import os
import requests
import base64
from openai import OpenAI
from starlette.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LINE API Credentials
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_ACCESS_TOKEN")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def verify_signature(request_body: str, signature: str) -> bool:
    """Verify that the request comes from LINE using HMAC SHA256."""
    hash_mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), request_body.encode("utf-8"), hashlib.sha256).digest()
    expected_signature = base64.b64encode(hash_mac).decode("utf-8")
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

                # Get ChatGPT response
                chatgpt_response = await get_chatgpt_response(user_message)

                # Reply to user
                await reply_to_user(reply_token, chatgpt_response)

    return JSONResponse(content={"status": "success", "message": "Webhook received"}, status_code=200)

async def get_chatgpt_response(user_message: str) -> str:
    """Send user message to ChatGPT and get a response."""
    try:
        response = openai_client.chat.completions.create(
            # chatgpt-4o-mint 
            model="gpt-4o-mini",  
            messages=[{"role": "system", "content": "You are Thai assistant a helpful and friendly bot  answers questions in Thai language."},
                      {"role": "user", "content": user_message}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return "I'm sorry, I couldn't process your request."

async def reply_to_user(reply_token: str, text: str):
    """Reply to user using LINE Messaging API."""
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


