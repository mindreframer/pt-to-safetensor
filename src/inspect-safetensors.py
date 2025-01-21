import safetensors
from safetensors import safe_open

# Path to your safetensors file
file_path = "yolov8s.safetensors"

# Method 1: Using `safe_open` as a context manager
with safe_open(file_path, framework="pt") as f:
    # Iterate through each tensor in the file
    for tensor_name in f.keys():
        tensor = f.get_tensor(tensor_name)

        # Print tensor name and shape
        print(f"Tensor Name: {tensor_name}")
        print(f"Shape: {tensor.shape}")
        print(f"Data Type: {tensor.dtype}")

        # Optionally, print the first few elements of the tensor
        print("First 5 elements:")
        print(tensor.flatten()[:5])

        print("-" * 40)
