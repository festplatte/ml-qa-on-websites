FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

COPY ./requirements.txt /workspace/
RUN conda install --y pip
RUN pip install  -r /workspace/requirements.txt

# Run a test if the container has all dependencies
RUN python -c "import torch"
RUN python -c "import transformers"
RUN python -c "import sentence_transformers"
RUN python -c "import jsonlines"
RUN python -c "import haystack"

WORKDIR /workspace
