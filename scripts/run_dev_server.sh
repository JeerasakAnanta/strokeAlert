#!/bin/bash 
echo "-------------- start dev server -------------------" 
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload