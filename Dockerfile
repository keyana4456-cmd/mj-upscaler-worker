# MJ Upscaler Runpod Serverless worker. Mirrors the Modal image in modal_runner.py.
# Stage 1 builds SageAttention 2.2 (pinned) against the exact torch/CUDA used at runtime, for Ada (4090, sm_89)
# and Hopper (H100/H200, sm_90). No GPU is needed to compile; the wheel is copied into the slim runtime image.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS sage
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv git && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /b && /b/bin/pip install --no-cache-dir torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu128 \
 && /b/bin/pip install --no-cache-dir wheel setuptools ninja packaging
RUN git clone https://github.com/thu-ml/SageAttention /src && cd /src && git checkout v2.2.0
# setup.py applies one arch list to every kernel, and the sm90 (wgmma) kernel can't compile for sm_89. So build twice:
#  A) 8.9+PTX -> sm80/sm89/fused kernels (+PTX so they also JIT on Hopper)   B) 9.0 -> keep only the sm90 kernel
ENV MAX_JOBS=3 EXT_PARALLEL=1 NVCC_APPEND_FLAGS="--threads 2"
RUN cp -r /src /srcA && cd /srcA && TORCH_CUDA_ARCH_LIST="8.9+PTX" /b/bin/python setup.py bdist_wheel && mkdir -p /out && cp dist/*.whl /out/
RUN cp -r /src /srcB && cd /srcB && TORCH_CUDA_ARCH_LIST="9.0" /b/bin/python setup.py build_ext --inplace \
 && cp sageattention/_qattn_sm90*.so /out/ && ls -l /out

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.11.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir numpy opencv-python-headless einops "omegaconf>=2.3.0" "diffusers>=0.33.1" "peft>=0.17.0" \
      "rotary_embedding_torch>=0.5.3" safetensors gguf tqdm psutil matplotlib huggingface_hub scipy scikit-image addict \
      imageio-ffmpeg av pyyaml requests timm yapf future onnxruntime-gpu==1.22.0 pillow runpod
COPY --from=sage /out/ /tmp/sage/
RUN pip install --no-cache-dir --no-deps /tmp/sage/sageattention-*.whl \
 && cp /tmp/sage/_qattn_sm90*.so "$(python -c 'import sageattention,os;print(os.path.dirname(sageattention.__file__))')/" \
 && python -c "import sageattention; print('sageattention ok')" && rm -rf /tmp/sage
RUN git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler /opt/mju/seedvr2 \
 && git clone --depth 1 https://github.com/sczhou/ProPainter /opt/mju/propainter \
 && rm -rf /opt/mju/propainter/weights && ln -s /runpod-volume/models/propainter_weights /opt/mju/propainter/weights \
 && ln -s /runpod-volume/models /opt/mju/models && ln -s /runpod-volume/models /models && ln -s /runpod-volume/io /io && ln -s /runpod-volume/io/work /opt/mju/work
ENV MJU_FFMPEG=ffmpeg MJU_PY=python MJU_WORK=/io/work PYTHONUTF8=1 HF_HOME=/runpod-volume/models/hf PYTHONUNBUFFERED=1
# Model weights live on the network volume under models/ (SeedVR2, CodeFormer, YuNet, SFace, text detector).
COPY app/ /opt/mju/
COPY handler.py /opt/mju/handler.py
WORKDIR /opt/mju
CMD ["python", "-u", "/opt/mju/handler.py"]
