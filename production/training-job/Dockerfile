FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7

# Install basics
RUN \
    apt-get update && \
    apt-get install sudo --yes && \
    apt-get install --yes --no-install-recommends  \
        ca-certificates \
        curl libcurl3 \
<<<<<<< HEAD
        libgl1-mesa-glx \ 
        software-properties-common \
=======
        zlibc zlib1g zlib1g-dev \
        software-properties-common \
        git \
>>>>>>> ae0f38ff81228459faa5187a1b29975f23ebbb28
        locales \
        gcc && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    chmod -R a+rwx /usr/local/bin/ && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/* && \
    rm -rf /tmp/*

# Install Miniconda Python 3.6 & PyTorch cuda10
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
<<<<<<< HEAD
    pip install typing && \
=======
    pip install --upgrade Cython && \
>>>>>>> ae0f38ff81228459faa5187a1b29975f23ebbb28
    conda clean -ya

# Install requirements
COPY docker-res/requirements.txt "/tmp/"
RUN pip install -r "tmp/requirements.txt"

# # Set env variables: paths for resources and data + some configs
ENV RESOURCES_PATH="/resources" \ 
    DATA_PATH="/data" \
    NUM_TRAIN_SAMPLES="5000" \
    NUM_TEST_SAMPLES="5000" \ 
    NUM_MAX_POSITIONS="256" \ 
    N_EPOCHS="2" \ 
    BATCH_SIZE="32" \
    VALID_PCT="0.05" \
    LR="0.000065" \
    MAX_NORM="1.5" \ 
    SEED="1337" \
    NVIDIA_VISIBLE_DEVICES="all" \
    OMP_NUM_THREADS="8" 

     
RUN \
    mkdir -p $RESOURCES_PATH && \
    mkdir -p $DATA_PATH && \
    chmod -R a+rwx $DATA_PATH

<<<<<<< HEAD
# copy resources
COPY docker-res/utils.py "$RESOURCES_PATH/"
COPY docker-res/run.py "$RESOURCES_PATH/"
CMD ["python", "/resources/run.py"]
=======
# Install Training Requirements


# Run job
COPY docker-res/run.py "$_RESOURCES_PATH/"
CMD ["python", "/resources/run.py"]
>>>>>>> ae0f38ff81228459faa5187a1b29975f23ebbb28
