# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/workspace/tmp \
    HF_HOME=/workspace/cache \
    TRITON_CACHE_DIR=/workspace/tmp/triton_cache

# ✅ Ensure temp and cache directories exist **before** installing packages
# RUN mkdir -p /workspace/tmp /workspace/cache && chmod -R 777 /workspace/tmp /workspace/cache

# ✅ Ensure temp, cache, and triton_cache directories exist before installing packages
# ✅ Ensure temp, cache, and Triton cache dirs exist at build time
# ✅ Ensure temp, cache, and Triton cache dirs exist at build time
RUN mkdir -p /workspace/tmp /workspace/cache "$TRITON_CACHE_DIR" \
    && chmod -R 777 /workspace/tmp /workspace/cache "$TRITON_CACHE_DIR"

# Fix missing APT repository issues
RUN apt-get update --fix-missing && apt-get install -y software-properties-common

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev \
    openjdk-11-jre-headless \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates

# Remove unnecessary cache files only after successful installation
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a virtual environment
WORKDIR /workspace
RUN python -m venv venv
ENV PATH="/workspace/venv/bin:$PATH"

# Install required Python libraries with fixed versions
RUN pip install --upgrade pip
RUN pip install \
    bitsandbytes==0.43.2 \
    accelerate==1.3.0 \
    peft==0.8.2 \
    #transformers==4.48.2 \
    #transformers==4.51.0 \
    transformers==4.53.2 \
    trl==0.14.0 \
    datasets==3.2.0 \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    tqdm==4.67.1 \
    numpy==1.23.5 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    rouge-score \
    nltk \
    pandas \ 
    xformers \
    deepspeed

#new one

# ✅ Ensure temp and cache directories exist again
RUN mkdir -p /workspace/tmp /workspace/cache && chmod -R 777 /workspace/tmp /workspace/cache

# ✅ Download PlantUML for UML validation
RUN wget https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar -O /workspace/plantuml.jar

# Copy the evaluation script and dataset
COPY evaluate_fewshots_feature1.py /workspace/
COPY evaluate_fewshots_feature2.py /workspace/
COPY evaluate_fewshots_feature3.py /workspace/
COPY evaluate_fewshots_feature4.py /workspace/
COPY evaluate_fewshots_feature5.py /workspace/
COPY evaluate_fewshots_inreal1.py /workspace/
COPY evaluate_fewshots_inreal2.py /workspace/
COPY evaluate_fewshots_inreal3.py /workspace/
COPY evaluate_fewshots_inreal4.py /workspace/
COPY few-shot_verification_dataset_20250406.json /workspace/
COPY few-shot_in_real_verification_dataset_20250424.json /workspace/
COPY RAG_raw_dataset_20250514.json /workspace/

# Set default command to run the evaluation script
# CMD ["python", "/workspace/evaluate_v22.py"]
COPY entrypoint.sh /workspace/

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]