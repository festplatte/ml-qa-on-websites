from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import jsonlines
import json
import re
import sys
import torch
import numpy as np
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever, EmbeddingRetriever

def main(args):
    model_path = args[1]
    elasticsearch_index = args[2]
    flags = args[3] if len(args) > 3 else ""

    document_store = ElasticsearchDocumentStore(host="localhost" if "l" in flags else "elasticsearch", index=elasticsearch_index, create_index=False, embedding_field="embedding")
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model=model_path, use_gpu=True, model_format="sentence_transformers") if "e" in flags else DensePassageRetriever(document_store=document_store, passage_embedding_model=model_path, use_gpu=True, embed_title=False)

    document_store.update_embeddings(retriever)

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        main(sys.argv)
    else:
        print("Usage: update-elasticsearch-embeddings.py <model path> <elasticsearch index> <flags>")
