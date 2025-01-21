import os
from safetensors import safe_open

def inspect_safetensors(file_path):
    # Create directories if they don't exist
    os.makedirs('tmp', exist_ok=True)
    log_base_file = file_path.replace('/', "__") + ".log"
    log_path = os.path.join('tmp', log_base_file)

    with safe_open(file_path, framework="pt") as f, open(log_path, 'w') as log:
        # Iterate through each tensor in the file
        for tensor_name in f.keys():
            tensor = f.get_tensor(tensor_name)

            # Write tensor name and shape to file
            log.write(f"Tensor Name: {tensor_name}\n")
            log.write(f"Shape: {tensor.shape}\n")
            log.write(f"Data Type: {tensor.dtype}\n")

            # Write first few elements of the tensor
            log.write("First 5 elements:\n")
            log.write(f"{tensor.flatten()[:5]}\n")

            log.write("-" * 40 + "\n")



inspect_safetensors("models_ref/yolov8s.safetensors")
inspect_safetensors("models_out/yolov8s.safetensors")
