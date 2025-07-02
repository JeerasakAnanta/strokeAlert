# Fastapi
from fastapi import APIRouter
from pydantic import BaseModel

# RAG system
from src.rag_system.sql_agent import sql_agent
from src.rag_system.sql_agent import assistant_agent

# LINE messaging API
from src.messaging_api.line_message_api import line_chatbot

# routers
router = APIRouter(prefix="/api")


class MessageRequest(BaseModel):
    user_message: str


# Routers
@router.get("/")
async def root():
    """
    Root API endpoint.

    Returns a JSON message indicating that the API is running.
    """
    return {"massge:": "Chatobt API is running...."}


@router.post("/ncd_agent")
async def get_ncd_response(request: MessageRequest) -> str:
    sql_answer = sql_agent(request.user_message)
    assistant_answer = assistant_agent(sql_answer)
    return assistant_answer


@router.post("/assistant_agent")
async def get_chatgpt_response(request: MessageRequest) -> str:

    return assistant_agent(request.user_message)


@router.post("/sql_agent")
async def get_sqlchat_response(request: MessageRequest) -> str:

    return sql_agent(request.user_message)


@router.post("/webhook")
async def line_webhook(request: MessageRequest):
    return line_chatbot(request.user_message)
