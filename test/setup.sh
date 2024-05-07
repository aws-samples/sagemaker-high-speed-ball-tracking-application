#!/bin/bash

pip install -r requirements.txt
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# sudo yum install mesa-libGL -y
if command -v yum &> /dev/null; then
    sudo yum install mesa-libGL -y
elif command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
    sudo apt-get install libgl1-mesa-dri -y
else
    echo "Neither apt nor yum package managers are available."
    exit 1
fi