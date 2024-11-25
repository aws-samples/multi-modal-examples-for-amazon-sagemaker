import uvicorn
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
import json
import asyncio
import logging
from typing import AsyncGenerator, Any
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

app = FastAPI()
predictor = None

logger = logging.getLogger("foo")

# Environment variables
model_id = os.environ.get('HF_MODEL_ID', "Qwen/Qwen2-VL-7B-Instruct")
dtype = os.environ.get('OPTION_DTYPE', "half")
max_model_len = int(os.environ.get('OPTION_MAX_MODEL_LEN', 8192))
gpu_memory_utilization = os.environ.get('OPTION_GPU_MEMORY_UTILIZATION', 0.95)
enforce_eager = eval(os.environ.get('OPTION_ENFORCE_EAGER', "False"))


# Initialize the model
def get_model():
    llm_model = LLM(
        model=model_id,
        limit_mm_per_prompt={
            "image": 2, 
            "video": 1
        },
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager
    )

    processor = AutoProcessor.from_pretrained(
        model_id
    )
    
    return llm_model, processor


@app.on_event("startup")
async def startup_event():
    global predictor, processor
    if not predictor:
        predictor, processor = get_model()


@app.get("/ping")
async def health_check():
    """Basic health check endpoint."""
    return {"Status": "Alive"}


@app.post("/invocations")
async def generate_text(request: Request) -> JSONResponse:
    global predictor, processor 

    try:
        request_dict = await request.json()
        logger.error(f"Received messages: {request_dict}, type: {type(request_dict)}")
        
        messages = request_dict.pop("messages")
        if "properties" in request_dict.keys():
            properties = request_dict.pop("properties")
        else:
            properties = {
                "temperature": 0.1,
                "top_p": 0.001,
                "repetition_penalty": 1.05,
                "max_tokens": 256,
                "stop_token_ids": []
            }
        
        sampling_params = SamplingParams(**properties)
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
        image_inputs, video_inputs = process_vision_info(messages)
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        
        # Generate outputs
        outputs = predictor.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        ret = {"text": generated_text, "prompt": prompt}

        logger.error(f"outputs: {ret}")
    
        return JSONResponse(
            status_code=200,
            content=ret
        )
        
    except Exception as e:
        # Capture the full traceback and convert it to a string
        error_traceback = traceback.format_exc()
        
        # Create a JSON response with the error details
        return JSONResponse(
            status_code=500,
            content={
                "text": None, 
                "prompt": None,
                "error": str(e),
                "traceback": error_traceback
            }
        )


if __name__ == "__main__":
    # Use uvicorn to run the FastAPI app with specified host and port
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
    
