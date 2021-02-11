from haystack.generator.base import BaseGenerator
from answerers import find_answerer

class HaystackGenerator(BaseGenerator):
    def __init__(self, model_path, cache_dir=None):
        AnswererClass = find_answerer(model_path)
        self.answerer = AnswererClass(model_path, cache_dir=cache_dir)

    def map_to_text(self, documents):
        result = []
        for doc in documents:
            result.append(doc.text)
        return result

    def predict(self, query, documents):
        answer = self.answerer.generate_answer(query, self.map_to_text(documents))
        return { "answer": answer }
