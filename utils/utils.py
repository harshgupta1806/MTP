import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import os

def load_image(img_path, size : tuple = (224, 224)):
    """Load and preprocess an image."""
    img = Image.open(img_path).resize(size)
    img_array = np.array(img, dtype=np.uint8)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# def run_inference(input_data, interpreter, num_iterations=1000):
#     """Run inference multiple times and measure total time."""
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Set the tensor once as the input doesn't change
#     interpreter.set_tensor(input_details[0]['index'], input_data)
    
#     start_time = time.time()
#     for _ in tqdm(range(num_iterations), desc="Running inferences"):
#         interpreter.invoke()
#     end_time = time.time()

#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     total_inference_time = end_time - start_time
#     average_time_per_inference = total_inference_time / num_iterations

#     return np.argmax(output_data), total_inference_time, average_time_per_inference

def run_inference(image_path, interpreter, num_iterations=1000, image_shape : tuple = (224, 224)):
    """Run inference on single image or multiple images from a folder."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(os.path.isdir(image_path))
    if os.path.isdir(image_path):  # If it's a directory, iterate over images
        total_inference_time = 0
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        print(image_files[:10])
        for img_file in tqdm(image_files, desc="Running inferences on folder"):
            img_full_path = os.path.join(image_path, img_file)
            input_data = load_image(img_full_path, image_shape)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference for a single iteration per image
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            total_inference_time += (end_time - start_time)
            
            # prediction = np.argmax(output_data)
            # print(f"Prediction for {img_file}: {prediction}")
        
        average_time_per_inference = total_inference_time / len(image_files)
        print(f"Total inference time for folder: {total_inference_time:.4f} seconds")
        print(f"Average time per image inference: {average_time_per_inference:.4f} seconds")
    
    else:  # If it's a single image, use the original behavior
        input_data = load_image(image_path, image_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        start_time = time.time()
        for _ in tqdm(range(num_iterations), desc="Running inferences on single image"):
            interpreter.invoke()
        end_time = time.time()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        total_inference_time = end_time - start_time
        average_time_per_inference = total_inference_time / num_iterations
        
        return np.argmax(output_data), total_inference_time, average_time_per_inference

def load_configs(*file_paths):
    merged_config = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file) or {}
            merged_config.update(config)
    return merged_config
