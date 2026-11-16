"""API دستیار:  uvicorn api:app --port 8000
مستندات تعاملی در /docs و رابط Gradio در /ui"""
import os
import tempfile
import uuid
from typing import Optional

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import agent
import database as db
from app import CSS, create_app

MAX_IMAGE_BYTES = 8 * 1024 * 1024

# ponytail: بدون احراز هویت کلاینت و فقط برای localhost؛ session_id حکم رمز جلسه را دارد.
# قبل از انتشار روی اینترنت، کلید API در هدر و محدودیت نرخ اضافه شود.
app = FastAPI(title="Sabzdasht Assistant API", version="1.0.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, min_length=16, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    trace: dict
    verified: bool


def _respond(session_id, reply, trace):
    return ChatResponse(session_id=session_id, reply=reply, trace=trace, verified=agent.get_citizen(session_id) is not None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or uuid.uuid4().hex
    try:
        reply, trace = agent.chat(session_id, request.message.strip())
    except Exception as e:
        raise HTTPException(502, f"model call failed: {e}")
    return _respond(session_id, reply, trace)


@app.post("/verify", response_model=ChatResponse)
def verify(session_id: str = Form(min_length=16, max_length=64), image: UploadFile = File(...)):
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(415, "file must be an image")
    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image is larger than 8 MB")

    suffix = "." + image.content_type.split("/")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
    try:
        reply, trace = agent.verify(session_id, f.name)
    finally:
        os.remove(f.name)
    return _respond(session_id, reply, trace)


@app.get("/products")
def list_products(q: str = "", category: str = ""):
    return db.search_products(q, category)


@app.get("/products/{sku}")
def product_detail(sku: str):
    product = db.get_product(sku)
    if not product:
        raise HTTPException(404, "product not found")
    return product


@app.get("/categories")
def categories():
    return db.list_categories()


app = gr.mount_gradio_app(app, create_app(), path="/ui", theme=gr.themes.Soft(), css=CSS)
