# Stroke Alert Chatbot Prototype

##  install 

- install  poetry 
```bash 
sudo apt install python3-poetry
```

-  install dependencies with poetry 
```bash
poetry install  
``` 

- use poetry enronment 
```bash
poetry shell
``` 

- run webhook server 
```bash
python3 -m webhook.py
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
