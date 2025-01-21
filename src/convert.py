import torch
from safetensors.torch import save_file


# torch.serialization.add_safe_globals([DetectionModel])

def main():
    # Load the model's state_dict
    state_dict = torch.load('yolov8n-doclaynet.pt', map_location='cpu', weights_only=False)

    # Save the state_dict to a safetensors file
    save_file(state_dict, 'yolov8n-doclaynet.safetensors')
    #
    # torch.save(state_dict, 'yolov8n-doclaynet.safetensors', _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    main()
