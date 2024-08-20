# Use the official Python image as a parent image
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

##### 
## Image Configuration
#####

ENV PYTHON_VERSION=3.12.4

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV LC_ALL C.UTF-8


# switch default shell from /bin/sh to /bin/bash to be able to use source
SHELL ["/bin/bash", "-c"]

##### 
## Tool Setup
#####

# Install dev dependencies

RUN apt update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get install -y git locales time gfortran

# Install python
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y --no-install-recommends wget gradle software-properties-common \
    build-essential libgraphviz-dev tar zlib1g-dev libffi-dev libreadline-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
    p7zip-full wget git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar -xf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make install && \
    ln -s /usr/local/bin/python3 /usr/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip


##### 
## Application Setup
#####

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python -

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app
COPY . /app

# # Install project dependencies using Poetry
# RUN poetry install --no-root

# # Command to run the application
# CMD ["echo", "hello"]
# CMD ["poetry", "run", "python", "-m", "model2regex"]