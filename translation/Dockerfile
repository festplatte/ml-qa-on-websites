FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY ./requirements.txt /workspace/
RUN conda install --y pip
RUN pip install  -r /workspace/requirements.txt

# Run a test if the container has all dependencies
RUN python -c "import torch"
RUN python -c "import transformers"
RUN python -c "import jsonlines"

WORKDIR /workspace
