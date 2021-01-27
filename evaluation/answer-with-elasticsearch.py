from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import jsonlines
import json
import re
import sys
import torch
import numpy as np
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever, EmbeddingRetriever


from answerers import find_answerer

torch.manual_seed(42)
np.random.seed(42)
# torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'

cache_dir = '/data/.cache'
save_after = 1000

# if torch.cuda.is_available():
#     model.cuda()

def get_elasticsearch_documents(es_retriever, query, top_k=1, probability_border=0.3, filter=False):
    result = []
    last_probability = 100
    documents = es_retriever.retrieve(query=query, top_k=top_k*5 if filter else top_k)
    if filter:
        for document in documents:
            if len(result) >= top_k or document.probability < probability_border:
                break
            if document.probability < last_probability:
                last_probability = document.probability
                result.append(document)
    else:
        for document in documents:
            result.append(document)
    return result

def get_file_lines(file):
    try:
        with open(file, 'r') as file1:
            lines = file1.readlines()
            return len(lines)
    except IOError:
        return 0

def map_to_ms_marco_passages(documents):
    result = []
    for doc in documents:
        result.append({
            "passage_text": doc.text,
            "is_selected": 1,
            "url": doc.meta['name'],
        })
    return result

def map_to_text(documents):
    result = []
    for doc in documents:
        result.append(doc.text)
    return result

def generate_answers(source_path, target_path, answerer, es_retriever, flags=""):
    source_content = ''
    with open(source_path, 'r') as source_file:
        source_content = source_file.read()

    source_file = source_content.splitlines()
    num_target_lines = get_file_lines(target_path)
    source_file = source_file[num_target_lines:]

    top_k = 1
    top_k = 10 if "m" in flags else top_k
    top_k = 20 if "b" in flags else top_k

    with jsonlines.open(target_path, "a") as target_file:
        processed_lines = []
        for line in tqdm(source_file, initial=num_target_lines, total=len(source_file) + num_target_lines):
            line_json = json.loads(line)
            documents = get_elasticsearch_documents(es_retriever, line_json['query'], top_k=top_k, filter="f" in flags)
            context = map_to_text(documents)
            line_json['generated_answer'] = answerer.generate_answer(line_json['query'], context)

            # save results
            processed_lines.append({
                "query_id": line_json["query_id"],
                "answers": [line_json["generated_answer"]],
                "reference_answers": line_json["answers"],
                "passages": map_to_ms_marco_passages(documents)
            })

            # print results
            print('documents:')
            print(documents)
            print('query: ' + line_json['query'])
            print('generated answer: ' + line_json['generated_answer'])

            # save to file
            if len(processed_lines) >= save_after:
                print('saving')
                target_file.write_all(processed_lines)
                processed_lines = []
        if len(processed_lines) > 0:
            target_file.write_all(processed_lines)

def main(args):
    source_path = args[1]
    target_path = args[2]
    model_path = args[3]
    elasticsearch_index = args[4]
    flags = args[5] if len(args) > 5 else ""
    AnswererClass = find_answerer(model_path)
    answerer = AnswererClass(model_path, cache_dir=cache_dir if "c" in flags else None)

    document_store = ElasticsearchDocumentStore(host="localhost" if "l" in flags else "elasticsearch", index=elasticsearch_index, create_index=False, embedding_field="embedding")
    retriever = ElasticsearchRetriever(document_store=document_store)
    if "d" in flags:
        retriever = DensePassageRetriever(document_store=document_store)
    if "e" in flags:
        retriever = EmbeddingRetriever(document_store=document_store, embedding_model="T-Systems-onsite/cross-en-de-roberta-sentence-transformer", use_gpu=False, model_format="sentence_transformers")

    generate_answers(source_path, target_path, answerer, retriever, flags)


if __name__ == '__main__':
    if len(sys.argv) >= 5:
        main(sys.argv)
    else:
        print("Usage: answer-with-elasticsearch.py <input file> <desired outputname> <model path> <elasticsearch index> <flags>")
