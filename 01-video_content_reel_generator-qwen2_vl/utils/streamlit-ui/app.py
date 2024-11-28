import numpy as np
import streamlit as st
from datetime import datetime
import json
import re
import copy
import subprocess
from typing import Dict, List
from PIL import Image  
import decord
import warnings
import io
import base64
import os
import boto3
import torch
import sagemaker
from sagemaker import serializers, deserializers
from decord import VideoReader, cpu
import requests

warnings.filterwarnings("ignore")

GLOBAL_TEMPERATURE = 0.1
GLOBAL_TOP_P = 0.001
REPTITION_PENALTY = 1.05
MAX_TOKENS = 8192
TOTAL_FRAMES = 8
app_path = os.path.dirname(os.path.realpath(__file__))

# model args
video_llm_model_id = "Qwen/Qwen2-VL-7B-Instruct"
endpoint_name = "qwen2vl-7b-instruct-endpoint"
headers = {"Content-Type": "application/json"}

st.set_page_config(page_title="vLLM with Qwen2-VL", page_icon="🤖")
st.image(os.path.join(app_path, "banner-large.svg"), use_container_width=True) 
st.title("🤖 SageMaker Multi-Modal Workshop ")

pretrained_predictor = sagemaker.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker.Session(),
    serializer=serializers.JSONSerializer(),
    deserializer=deserializers.JSONDeserializer(),
)

conversation_history = [
    {
        "role": "system", 
        "content": "You are a helpful assistant. You will prioritize the conversation between a human and an assistant and respond polietly."
    }
]

# Sidebar Configuration
st.sidebar.image(os.path.join(app_path, "banner-small.svg"))
st.sidebar.title("Configuration:")
st.sidebar.subheader("Basic Settings:")

# Temperature Slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=GLOBAL_TEMPERATURE,
    step=0.1
)

# Top_p Slider
top_p = st.sidebar.slider(
    "Top p",
    min_value=0.001,
    max_value=1.0,
    value=GLOBAL_TOP_P,
    step=0.05
)

# Max Tokens Slider
max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=1024,
    max_value=10240,
    value=MAX_TOKENS,
    step=1024
)

# Advanced Model Settings
st.sidebar.subheader("Advanced Settings:")

# Stop Words Input
stop_words = st.sidebar.text_input(
    "Stop Token IDs",
    value="[]"
)

# Frequency Penalty Slider
session_frames = st.sidebar.slider(
    "Video Frames to Sub-Sample",
    min_value=TOTAL_FRAMES,
    max_value=12,
    value=TOTAL_FRAMES,
    step=1
)

# Frequency Penalty Slider
repetition_penalty = st.sidebar.slider(
    "Repetition Penalty",
    min_value=0.5,
    max_value=2.0,
    value=REPTITION_PENALTY,
    step=0.5
)

# Model Selection
model_choice = st.sidebar.selectbox(
    "Choose Model",
    (f"{video_llm_model_id}")
)

# Prompt Template Selection
prompt_template = st.sidebar.selectbox(
    "Prompt Template",
    ("system/user/assistant")
)

# Initialize chat history and media display flag in session state if they do not exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = copy.deepcopy(conversation_history)
if "media" not in st.session_state:
    st.session_state.media = None
if "media_displayed" not in st.session_state:
    st.session_state.media_displayed = False

# Container for Upload Box (fixed at the top)
with st.container():
    st.subheader("Upload Media (Image or Video)")
    uploaded_file = st.file_uploader("Choose an image or video file...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

    # Clear Chat Button
    if st.button("Clear Chat"):
        st.session_state.messages.clear()
        st.session_state.history = copy.deepcopy(conversation_history)
        st.session_state.media = None  # Clear stored media
        st.session_state.media_displayed = False  # Reset media display flag
        system_chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

# Process the uploaded file only if it's a new upload
if uploaded_file is not None and not st.session_state.media_displayed:
    file_type = uploaded_file.type
    if "image" in file_type:
        # Handle image
        image = Image.open(uploaded_file)
        # Store in session state
        st.session_state.media = image
    elif "video" in file_type:
        # Handle video
        video_bytes = uploaded_file.read()  # Read video bytes to pass to st.video()
        # Store in session state
        st.session_state.media = video_bytes
    st.session_state.media_displayed = True  # Set flag to prevent duplicate display

# Display the media only once, immediately after upload
if st.session_state.media is not None and st.session_state.media_displayed:
    if isinstance(st.session_state.media, Image.Image):
        st.image(st.session_state.media, caption="Uploaded Image", use_container_width=True)
    else:
        st.video(st.session_state.media)

# Display chat messages from history without repeating media display
avatars = {"user": "human", "assistant": "ai"}
for msg in st.session_state.messages:
    with st.chat_message(avatars[msg["type"]]):
        st.write(msg["content"])


def encode_image(media):
    media = media.resize((300, 300), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    media.save(buffered, format="PNG")

    # Base64 encode the byte stream
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")

    # Create a data URI for the base64 image
    base64_str = f"data:image/png;base64,{encoded_image_text}"

    return base64_str


# Input for new message
if prompt := st.chat_input(placeholder="Please ask me a question!"):
    
    # Add user message to history
    st.session_state.messages.append({"type": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if st.session_state.media is not None:
        if isinstance(st.session_state.media, Image.Image):
            base64_enc_str = encode_image(st.session_state.media)
            st.session_state.history.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_enc_str},
                    {"type": "text", "text": prompt}
                ]
            })
        else:
            _video_bytes = st.session_state.media
            vr = VideoReader(io.BytesIO(_video_bytes), ctx=cpu(0))
            total_frames = len(vr)
            print("total_frames ===>", total_frames, "selecting: ", session_frames)
            selected_frames = np.linspace(0, total_frames - 1, session_frames, dtype=int)
            print("selected frames ===>", selected_frames)
            video_frames = vr.get_batch(selected_frames).asnumpy()
            video_base64_as_list = [encode_image(Image.fromarray(x)) for x in video_frames]
            st.session_state.history.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": video_base64_as_list, "fps": 1.0},
                    {"type": "text", "text": prompt}
                ]
            })
    else:
        st.session_state.history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        })

    # Get assistant response
    with st.chat_message("assistant"):
        payload = {
            "messages": st.session_state.history,
            "properties": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
                "stop_token_ids": []
            }
        }
        response = pretrained_predictor.predict(payload)
        generated_text = response["text"]
        st.write(generated_text)
        # Add assistant response to history
        st.session_state.history.append({
            "role": "assistant",
            "content": generated_text
        })
        st.session_state.messages.append({"type": "assistant", "content": generated_text})
