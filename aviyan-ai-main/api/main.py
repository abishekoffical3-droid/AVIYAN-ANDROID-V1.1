from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from aviyan_core.runtime import AviyanRuntime

app = FastAPI(title="AVIYAN API", version="1.1.0")
runtime = AviyanRuntime()


class Msg(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "aviyan-5b"
    messages: list[Msg] = Field(default_factory=list)
    user_id: str = "default"


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    duration_seconds: int = 5
    image_url: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "name": "AVIYAN",
        "creator": "ABISHEK BHUSAL",
        "company": "AB DEV STUDIO",
        "origin": "NEPAL",
        "version": "1.1.0",
    }


@app.get("/v1/identity")
def identity():
    return runtime.identity()


@app.post("/v1/chat")
def chat(r: ChatRequest):
    if not r.messages:
        content = "Hello. I am AVIYAN. How can I help you?"
    else:
        content = runtime.reply(r.user_id, r.messages[-1].content)
    return {"model": r.model, "choices": [{"message": {"role": "assistant", "content": content}}]}


@app.post("/v1/tools/calculator")
def calculator(body: dict):
    from aviyan_core.tools.calculator import calculate
    return {"result": calculate(body["expression"])}


@app.post("/v1/tools/python")
def python_tool(body: dict):
    # Development-only endpoint. Put it behind an authenticated sandbox in production.
    from aviyan_core.tools.python_exec import run_python
    return run_python(body["code"])


@app.post("/v1/generate/image")
def generate_image(r: GenerationRequest):
    return {
        "status": "backend_required",
        "type": "text-to-image",
        "prompt": r.prompt,
        "message": "Connect an image-generation backend/model to produce the image.",
    }


@app.post("/v1/generate/video")
def generate_video(r: GenerationRequest):
    return {
        "status": "backend_required",
        "type": "text-to-video" if not r.image_url else "image-to-video",
        "prompt": r.prompt,
        "message": "Connect a video-generation backend/model to produce the video.",
    }


@app.post("/v1/edit/image")
def edit_image(r: GenerationRequest):
    return {
        "status": "backend_required",
        "type": "image-edit",
        "prompt": r.prompt,
        "image_url": r.image_url,
        "message": "Connect an image-editing backend/model to apply the edit.",
    }
