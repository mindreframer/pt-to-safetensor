#!/usr/bin/env bash

# Create directories if they don't exist
mkdir -p models_ref models_in

if [ ! -f models_ref/yolov8s.safetensors ]; then
    echo "downloading reference model for comparison..."
    wget "https://huggingface.co/lmz/candle-yolo-v8/resolve/main/yolov8s.safetensors?download=true" -O models_ref/yolov8s.safetensors
else
    echo "reference model already exists, skipping download"
fi

if [ ! -f models_in/yolov8s.pt ]; then
    echo "downloading model for conversion..."
    wget "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt" -O models_in/yolov8s.pt
else
    echo "conversion model already exists, skipping download"
fi
