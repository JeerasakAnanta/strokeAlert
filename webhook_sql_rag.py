# dev by jeerasak anant
import os

# FastAPI server
from fastapi import FastAPI, Request, HTTPException
import uvicorn

# logging
import logging

# LINE Messaging API
import json
import hmac
import hashlib
import requests
import base64
from starlette.responses import JSONResponse

from openai import OpenAI

# langchain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()


# Constants for LINE API
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_ACCESS_TOKEN"
)

# Constants for MySQL Database
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_host = os.getenv("MYSQL_HOST")
mysql_port = os.getenv("MYSQL_PORT")
mysql_db = os.getenv("MYSQL_DATABASE")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ==================== RAG_SQL Agent ====================
def sql_agent(user_input: str) -> str:
    db_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"

    try:
        db = SQLDatabase.from_uri(db_uri)
    except Exception as e:
        return f"[ERROR] Failed to connect to the database: {str(e)}"

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
        return agent_executor.run(user_input)
    except Exception as e:
        return f"[ERROR] {str(e)}"


def assistant_agent(sql_input: str) -> str:
    """
    This function generates a response from the assistant agent
    based on the SQL query results provided as input.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """คุณคือผู้ช่วย อาสาสมัครสาธารณสุขประจำหมู่บ้าน (อสม.) 
        ข้อมูลสุขภาพของผู้ป่วย  คุณสามารถตอบคำถามเกี่ยวกับข้อมูลสุขภาพของผู้ป่วยได้ โดยข้อมูลมี Field อยู่ในฐานข้อมูลมีดังนี้  
        hn               -- หมายเลขผู้ป่วย,
        date             -- วันที่บันทึกข้อมูล
        full_name        -- ชื่อและนามสกุลของผู้ป่วย
        address          -- ที่อยู่ของผู้ป่วย
        bmi              -- ดัชนีมวลกาย (Body Mass Index)
        sbp              -- ความดันโลหิตซิสโตลิก (Systolic Blood Pressure)
        dbp              -- ความดันโลหิตด้านล่าง (Diastolic Blood Pressure)
        blood_sugar      -- ระดับน้ำตาลในเลือด
        bone_status      -- สถานะของกระดูก
        dementia         -- สถานะสมองเสื่อม     (0 = เท็จ, 1 = จริง)
        depression       -- สถานะซึมเศร้า       (0 = เท็จ, 1 = จริง)
        nutrition_status -- สถานะโภชนาการ
        smoking          -- สถานะการสูบบุหรี่     (0 = เท็จ, 1 = จริง).
        recorder         -- ผู้บันทึกข้อมูล    
        ตอบโดยใช้ภาษาไทย และ มี emoji ประกอบการตอบคำถาม
        ตัวอย่างการตอบคำถาม เช่น
        ✅ โปรดตอบโดยใช้ภาษาไทย และใส่ emoji ให้เหมาะสม เช่น
        - 🧑‍⚕️ ชื่อผู้ป่วย
        - 🏠 ที่อยู่
        - ⚖️ BMI
        - 💓 ความดันโลหิต
        - 🍬 น้ำตาลในเลือด
        - 🧠 สมองเสื่อม
        - 😔 ซึมเศร้า
        - 🚬 สูบบุหรี่
        - 📝 บันทึกล่าสุด

        ✅ ตัวอย่างคำตอบที่ดี:
        "🧑‍⚕️นาย ก ขาว  
        🏠 ที่อยู่: 88/8 หมู่ 5 ต.สุขใจ  
        💓 ความดัน: 120/80 mmHg  
        🍬 น้ำตาล: 95 mg/dL  
        ⚖️ BMI: 23.5  
        🧠 ไม่มีภาวะสมองเสื่อม  
        😔 ไม่มีภาวะซึมเศร้า  
        🚬 ไม่สูบบุหรี่  
        📝 บันทึกล่าสุด: 14 กุมภาพันธ์ 2568"
        ยินดีที่ได้ช่วยเหลือคุณในเรื่องข้อมูลสุขภาพของผู้ป่วยนะครับ 😊
        """,
            ),
            ("user", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm
    result_summary = chain.invoke({"input": f"สรุปข้อมูล  {sql_input}"})

    return result_summary.content


# ==================== LINE Messaging API ====================
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


@app.post("/webhook")
async def line_webhook(request: Request):
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


async def get_sqlchat_response(user_message: str) -> str:
    """
    Sends user messages to the ChatGPT model and retrieves a response.
    """
    try:
        sql_result = sql_agent(user_message)
        output_assistant_agent = assistant_agent(sql_result)
        return output_assistant_agent
    except Exception as e:
        logging.error(f"error: {str(e)}")
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

    if response.status_code == 200:
        logging.info("Reply sent successfully!")
    else:
        logging.error(f"Failed to send reply: {response.text}")


# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
