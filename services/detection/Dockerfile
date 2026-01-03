FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# install torch + ultralytics + other deps (use official cu118 wheels)
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio --extra-index-url https://pypi.org/simple
RUN python3 -m pip install --no-cache-dir ultralytics pika opencv-python-headless numpy

ENV PYTHONUNBUFFERED=1
CMD ["python3", "app.py"]
