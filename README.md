## Script to convert YOLO pt files to safetensors


### Requirements
- Python
- uv (https://docs.astral.sh/uv/)


### Usage

```bash

## download models
bin/download.sh

## activate virtual environment
$ uv venv

# run official convert script
$ uv run scr/convert-official.py

# run inspect pt script
$ uv run scr/inspect-pt.py

# run inspect safetensors script (will output log files to tmp/ folder)
$ uv run scr/inspect-safetensors.py
```


Reference:
- https://huggingface.co/lmz/candle-yolo-v8/tree/main
- https://huggingface.co/lmz/candle-yolo-v8/discussions/1
- https://github.com/ultralytics/assets/releases/tag/v0.0.0


### Files:
- https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
- https://huggingface.co/lmz/candle-yolo-v8/resolve/main/yolov8s.safetensors?download=true
