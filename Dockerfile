ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.8.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}


ENV TZ=Canada/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y --no-install-recommends \
    tzdata \
    git \
    build-essential \
    python3.9 \
    python3-pip \
    python3-setuptools \
    # locales \
    sudo
#     localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
# ENV LANG en_US.utf8

RUN pip3 -q install pip --upgrade

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENV WANDB_API_KEY="b49c726803eb9a4b39641921189be7d04cc3f9e1"
RUN pip install wandb --upgrade

# Create an user for the app.
# ARG USER=docker
# ARG PASSWORD=docker
# RUN useradd --shell /bin/bash --groups sudo ${USER}
# RUN echo ${USER}:${PASSWORD} | chpasswd
# USER ${USER}

# root user for dev branch
USER root

WORKDIR /Major_Project


CMD ["python3", "Deep_Learning_Image_Steganography/Stega/trainer.py"]
# ENTRYPOINT ["python3", "Deep_Learning_Image_Steganography/Stega/trainer.py", "1"] # Pass args
EXPOSE 9092