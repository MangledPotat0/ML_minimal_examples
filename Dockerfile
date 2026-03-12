FROM nvcr.io/nvidia/jax:25.08-py3

WORKDIR /app/workdir
COPY requirements.txt .

RUN apt-get update && apt-get install -y vim openjdk-17-jdk

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]
