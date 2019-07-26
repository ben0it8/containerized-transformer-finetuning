# Use nvidia image /w CUDA10 for GPU acceleration
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6

# Install basics
RUN \
    apt-get update && \
    apt-get install sudo --yes && \
    apt-get install --yes --no-install-recommends  \
        software-properties-common \
        build-essential \
        gcc \ 
        curl \
        locales && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    chmod -R a+rwx /usr/local/bin/ && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/* && \
    rm -rf /tmp/*

# Install Miniconda /w Python 3.6 
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    XDG_CACHE_HOME="/root/.cache/"

# Update python basics
RUN \
    conda install -y setuptools && \    
    pip install --upgrade pip && \
    pip install typing

# Install training and app requirements
COPY docker-res/requirements.txt "/tmp/"
RUN pip install -r "tmp/requirements.txt" && \
    pip install --upgrade bottle && \
    conda clean -ya

# Set env variables: paths for resources and data + some configs
ENV RESOURCES_PATH="/resources" \ 
    DATA_PATH="/data" \ 
    LOG_PATH="/logs" \ 
    NUM_TRAIN="5000" \
    NUM_TEST="5000" \ 
    NUM_MAX_POSITIONS="256" \ 
    N_EPOCHS="1" \ 
    BATCH_SIZE="8" \
    VALID_PCT="0.05" \
    LR="0.000065" \
    MAX_NORM="2.0" \ 
    SEED="1337" \
    NVIDIA_VISIBLE_DEVICES="all" \
    OMP_NUM_THREADS="8"  \
    BODY="1"
     
RUN \
    mkdir -p $RESOURCES_PATH && \
    mkdir -p $DATA_PATH && \
    chmod -R a+rwx $DATA_PATH

# Copy resources
COPY docker-res/utils.py "$RESOURCES_PATH/"
COPY docker-res/fit.py "$RESOURCES_PATH/"
COPY docker-res/app.py "$RESOURCES_PATH/"

# Run training and start webapp at localhost:5000
CMD ["sh","-c", "python /resources/fit.py && python /resources/app.py"]
