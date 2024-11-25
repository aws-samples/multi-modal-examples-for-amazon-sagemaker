FROM vllm/vllm-openai:v0.6.1

# Install required packages
RUN pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
RUN pip install sagemaker==2.231.0
RUN pip install qwen-vl-utils
RUN pip install qwen-vl-utils[decord]


# Copy the inference script to /app
COPY inference.py /app/inference.py

# Set working directory
WORKDIR /app

# Expose the port for uvicorn
EXPOSE 8080

ENTRYPOINT ["python3", "/app/inference.py"]