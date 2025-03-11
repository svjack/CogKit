"""
This script creates an OpenAI Server demo with transformers for the CogView4 model,
using the OpenAI API to interact with the model.

You can specify the model path, host, and port via command-line arguments, for example:
python openai_demo.py --model_path THUDM/CogView4-6B --host 0.0.0.0 --port 8000
"""

import gc
import time
import base64
import argparse
from pathlib import Path
from typing import list, Literal
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from diffusers import CogView4Pipeline


@asynccontextmanager
async def lifespan():
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential
    for efficient resource management in GPU environments.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

# Enable CORS so that the API can be called from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    """
    A Pydantic model representing a model card, which provides metadata about a machine learning model.
    It includes fields like model ID, owner, and creation time.
    """

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: str = None
    parent: str = None
    permission: list = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = TextContent | ImageUrlContent


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str | list[ContentItem]
    name: str = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: str = None


class DeltaMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = None
    content: str = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessageInput]
    temperature: float = 0.8
    top_p: float = 0.8
    max_tokens: int = None
    stream: bool = False
    repetition_penalty: float = 1.0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: list[ChatCompletionResponseChoice | ChatCompletionResponseStreamChoice]
    created: int = Field(default_factory=lambda: int(time.time()))
    usage: UsageInfo = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    An endpoint to list available models. It returns a list of model cards.
    This is useful for clients to query and understand what models are available for use.
    """
    model_card = ModelCard(id="CogView4")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_image_completion(request: ChatCompletionRequest):
    """
    An endpoint to create image completions given a set of messages and model parameters.
    Returns either a single completion or streams tokens as they are generated.
    """
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    messages = request.messages
    history = process_history(messages)
    response = generate_image(
        prompt=history,
        guidance_scale=3.5,
        num_images_per_prompt=1,
        num_inference_steps=50,
        width=1024,
        height=1024,
    )

    usage = UsageInfo()
    message = ChatMessageResponse(role="assistant", content=response["image"])
    choice_data = ChatCompletionResponseChoice(index=0, message=message)

    # task_usage = UsageInfo.model_validate(response["usage"])
    # for usage_key, usage_value in task_usage.model_dump().items():
    #     setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
        usage=usage,
    )


def process_history(messages: list[ChatMessageInput]) -> str:
    text_content = ""
    for message in messages:
        content = message.content

        # Extract text content
        if isinstance(content, list):
            extracted_texts = [item.text for item in content if isinstance(item, TextContent)]
            text_content = " ".join(extracted_texts)

        else:
            # If content is a string, treat it directly as text
            text_content = content
    return text_content


def encode_image(image_path):
    # Encodes an Image into a base64 string.
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def generate_image(
    prompt,
    guidance_scale,
    num_images_per_prompt,
    num_inference_steps,
    width,
    height,
):
    global pipe
    # Generate the image based on the prompt
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
    ).images[0]
    image.save("cogview4.png")
    img_url = encode_image("cogview4.png")

    return {
        "image": img_url,
        # "usage": {
        #     "prompt_tokens": input_echo_len,
        #     "completion_tokens": total_len - input_echo_len,
        #     "total_tokens": total_len,
        # },
    }


# Clean up GPU memory if possible
gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    # Use argparse to control model_path, host, and port from command line arguments
    parser = argparse.ArgumentParser(description="OpenAI Server Demo for CogView4")
    parser.add_argument("--model_path", required=True, help="Path or name of the CogView4 model")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    model_dir = Path(args.model_path).expanduser().resolve()

    # Load the pre-trained model with the specified precision
    pipe = CogView4Pipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

    # Enable CPU offloading to free up GPU memory when layers are not actively being used
    pipe.enable_model_cpu_offload()

    # Enable VAE slicing and tiling for memory optimization
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Run the Uvicorn server with the specified host and port
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
