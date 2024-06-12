from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List
from chatbot import ChatBot
from test import ConversationalRAG

app = FastAPI()
# chatbot = ChatBot()
links = [
    "https://huggingface.co/learn/nlp-course/chapter3/1?fw=pt",
    "https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt",
    "https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt",
    "https://huggingface.co/learn/nlp-course/chapter3/4?fw=pt",
    "https://huggingface.co/learn/nlp-course/chapter3/5?fw=pt",
    "https://huggingface.co/learn/nlp-course/chapter3/6?fw=pt",
]

conversational_rag = ConversationalRAG(links)
session_id = "3"
templates = Jinja2Templates(directory="templates")


class Message(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_history": conversational_rag.get_chat_history(session_id=session_id),
        },
    )


@app.post("/chat/", response_class=HTMLResponse)
async def post_chat(request: Request, message: str = Form(...)):
    response = conversational_rag.get_response(
        input_message=message, session_id=session_id
    )
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "response": response,
            "chat_history": conversational_rag.get_chat_history(session_id=session_id),
        },
    )


@app.get("/history/")
async def history():
    chat_history = conversational_rag.get_chat_history(session_id=session_id)
    return {"chat_history": chat_history}
