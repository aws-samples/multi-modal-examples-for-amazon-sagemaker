import subprocess
import sys
import os

def install_dependencies():
    """Install dependencies from requirements.txt"""
    try:
        # Get the directory where inference.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        requirements_path = os.path.join(current_dir, 'requirements.txt')
        
        if os.path.exists(requirements_path):
            print("Installing dependencies from requirements.txt...")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "-r", 
                requirements_path
            ])
            print("Dependencies installed successfully")
        else:
            print(f"requirements.txt not found in {current_dir}")
            
    except Exception as e:
        print(f"Error installing dependencies: {str(e)}")
        raise

def model_fn(model_dir):
    """Load model from model_dir."""
    print(f"loading model: {model_dir}")
    install_dependencies()
    print("installed dependencies")
    # Your model loading code here
    return model_dir

def input_fn(input_data, content_type):
    """Transform input data into format required by the model."""
    # Your input processing code here
    return input_data

def predict_fn(input_data, model):

    print(f"model dir: {model}")
    print(f"input data: {input_data}")

    from swift.llm import (
        InferArguments, ModelType, infer_main
    )


    import torch
    
    model_type = ModelType.qwen2_vl_2b_instruct

    
    torch.cuda.empty_cache()

    data_file = "./test_data.jsonl"
    with open(data_file, 'w') as f:
        f.write(input_data.decode('utf-8') + '\n')

    ckpt_dir = os.path.join(model,"qwen2-vl-2b-instruct/v0-20241111-130350/checkpoint-5")
    infer_args = InferArguments(
        model_type=model_type,
        ckpt_dir=ckpt_dir,
        result_dir='/opt/ml/processing/output',
        val_dataset=data_file
    )
    # merge_lora(infer_args, device_map='cpu')
    result = infer_main(infer_args)
    print(f"result {result}")
    torch.cuda.empty_cache()
   


    
   

def output_fn(prediction, accept):
    """Transform prediction into required output format."""
    print(f"prediction: {prediction}")
    return prediction