# Stroke Alert Chatbot Prototype

## Requirements  
- Python 
- Line Messaging API 
- (LLM) Openai API
- Fastapi
- Langchain 
- Docker 
- Ngrok 
- MySQL database 
- UV python package manager   

## Setup 
- clone repo
```bash  
https://github.com/JeerasakAnanta/strokeAlert.git
cd strokeAlert
```

### environment 

- install [uv](https://docs.astral.sh/uv/)  
```bash 
# linux and mac  
curl -LsSf https://astral.sh/uv/install.sh | sh
```

-  create  environment 
```bash
uv venv
source venv/bin/activate
``` 

- update  install package 
```bash
uv sync 
``` 

- run webhook server 
```bash
uv run webhook_rag.py
```  
- webhook server will run on port 8000 by default.
http://you-ip:8000/webhook


##  setup  ngrok with docker  

-  install docker 
```bash
docker run --net=host -it -e NGROK_AUTHTOKEN="YOUR_NGROK_TOKEN" ngrok/ngrok:latest http 8000
```
-  ngrok will run on port 8000 by default.
-  https://YOUR_NGROK.ngrok-free.app

