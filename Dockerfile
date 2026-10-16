# MJ Upscaler Runpod Serverless worker. Mirrors the Modal image in modal_runner.py.
# Build context = the staged folder made by runpod_setup.py --stage (contains app/ and this file).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.11.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir numpy opencv-python-headless einops "omegaconf>=2.3.0" "diffusers>=0.33.1" "peft>=0.17.0" \
      "rotary_embedding_torch>=0.5.3" safetensors gguf tqdm psutil matplotlib huggingface_hub scipy scikit-image addict \
      imageio-ffmpeg av pyyaml requests timm yapf future onnxruntime-gpu==1.22.0 pillow runpod
RUN git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler /opt/mju/seedvr2 \
 && git clone --depth 1 https://github.com/sczhou/ProPainter /opt/mju/propainter \
 && rm -rf /opt/mju/propainter/weights && ln -s /runpod-volume/models/propainter_weights /opt/mju/propainter/weights \
 && ln -s /runpod-volume/models /opt/mju/models && ln -s /runpod-volume/models /models && ln -s /runpod-volume/io /io && ln -s /runpod-volume/io/work /opt/mju/work
ENV MJU_FFMPEG=ffmpeg MJU_PY=python MJU_WORK=/io/work PYTHONUTF8=1 HF_HOME=/runpod-volume/models/hf PYTHONUNBUFFERED=1
# Model weights (SeedVR2 3B fp8 + VAE, CodeFormer, YuNet, text detector) live on the network volume under models/;
# runpod_setup.py uploads them from this PC once. They are not baked in, so the image stays ~8 GB.
COPY app/ /opt/mju/
COPY handler.py /opt/mju/handler.py
WORKDIR /opt/mju
CMD ["python", "-u", "/opt/mju/handler.py"]
