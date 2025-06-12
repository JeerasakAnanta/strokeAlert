# main.py
# dev by jeerasak anant

import os
import json
import hmac
import base64
import hashlib
import requests
import logging

from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import uvicorn

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# docling
from docling.document_converter import DocumentConverter

# ========== Load Environment Variables ==========
load_dotenv()
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_ACCESS_TOKEN"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========== FastAPI App ==========
app = FastAPI()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ========== Load & Process Dataset ==========
with open("dataset.md", "r", encoding="utf-8") as f:
    markdown_data = f.read()

content_list = markdown_data.split("\n")

langchain_documents = [
    Document(page_content=content.strip()) for content in content_list
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(langchain_documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 20})
llm = ChatOpenAI(model="gpt-4o-mini")

# ========== RAG Prompt ==========
template = """Answer the question based only on the following context:
คุณคือผู้ช่วย อสม. ข้อมูลผู้ป่วย:
- HN
- วันที่
- ชื่อ-สกุล
- ที่อยู่
- BMI
- SBP / DBP
- น้ำตาล
- กระดูก
- สมองเสื่อม
- ซึมเศร้า
- โภชนาการ
- บุหรี่
- ผู้บันทึก

โปรดตอบเป็นภาษาไทย พร้อม emoji 🤖
{context}

Question: {query}
"""
prompt = ChatPromptTemplate.from_template(template)
qa_chain = prompt | llm | StrOutputParser()


# ========== LINE Signature ==========
def verify_signature(request_body: str, signature: str) -> bool:
    hash_mac = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"),
        request_body.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    expected_signature = base64.b64encode(hash_mac).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


# ========== Format Docs ==========
def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


# ========== Webhook ==========
@app.post("/webhook")
async def line_webhook(request: Request):
    signature = request.headers.get("X-Line-Signature")
    request_body = await request.body()

    if not verify_signature(request_body.decode("utf-8"), signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    data = await request.json()
    logging.info(f"Received LINE webhook: {json.dumps(data, indent=2)}")

    for event in data.get("events", []):
        if event.get("type") == "message":
            user_id = event["source"]["userId"]
            reply_token = event["replyToken"]
            user_message = event["message"]["text"]

            response_text = await rag_query(user_message)
            await reply_to_user(reply_token, response_text)

    return JSONResponse(content={"status": "success"}, status_code=200)


# ========== RAG Query ==========
async def rag_query(query: str) -> str:
    try:
        relevant_docs = retriever.invoke(query)
        response = qa_chain.invoke(
            {"context": format_docs(relevant_docs), "query": query}
        )
        return response
    except Exception as e:
        logging.error(f"RAG error: {str(e)}")
        return "ขออภัยค่ะ ไม่สามารถประมวลผลคำถามได้ 😢"

# ========== Reply to LINE ==========
async def reply_to_user(reply_token: str, text: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}],
    }
    response = requests.post(
        "https://api.line.me/v2/bot/message/reply", json=payload, headers=headers
    )

    if response.status_code == 200:
        logging.info("Reply sent successfully!")
    else:
        logging.error(f"Failed to send reply: {response.text}")


# ========== Run Server ==========
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
