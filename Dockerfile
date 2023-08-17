
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

WORKDIR /Major_Project

RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3.9 \
    python3-pip \
    python3-setuptools

RUN pip3 -q install pip --upgrade

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


# CMD ["python3", "Stega/trainer.py"]
EXPOSE 9092