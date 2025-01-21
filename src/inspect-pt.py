import torch

# model_name = "models/doclayout_yolo_docstructbench_imgsz1024.pt"
model_name = "models/yolov8s.pt"

def main():
    data = torch.load(model_name, map_location='cpu', weights_only=False)

    if isinstance(data, torch.Tensor):
        print("Loaded a single tensor.")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"First 5 elements: {data.flatten()[:5]}")
    elif isinstance(data, dict):
        print("Loaded a dictionary of tensors and scalars.")
        for key, value in data.items():
            print(f"Key: {key}")
            if isinstance(value, torch.Tensor):
                print(f"  Type: Tensor")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  First 5 elements: {value.flatten()[:5]}")
            else:
                print(f"  Type: {type(value).__name__}")
                print(f"  Value: {value}")
            print("-" * 40)
    else:
        print("Loaded data is neither a tensor nor a dictionary.")
        print(f"Data type: {type(data).__name__}")
        print(f"Data: {data}")


if __name__ == "__main__":
    main()
