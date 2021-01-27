FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY ./requirements.txt /workspace/
RUN conda install --y pip
RUN pip install  -r /workspace/requirements.txt

# Run a test if the container has all dependencies
RUN python -c "import torch"
RUN python -c "import scrapy"
RUN python -c "import spacy"

# download spacy models
RUN python -m spacy download de_core_news_md
RUN python -m spacy download de_core_news_lg
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg

WORKDIR /workspace
