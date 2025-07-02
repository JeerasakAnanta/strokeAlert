# FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes import routers

import uvicorn

# heper function
from src.utils import check_db_connection

app = FastAPI(
    title="Strokealert Prototype",
    description="This is a prototype for Strokealert",
    version="0.3.0",
)

# Add CORS middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("Database connection status:", check_db_connection())
app.include_router(routers.router)
