from transformers import AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re
import torch
import os

class GPT2Answerer:
    """
    This class uses a gpt2 model to answer questions.
    """

    stop_token = '<EOS>'

    def __init__(self, model_path, cache_dir=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

    def build_model_input(self, query, passages):
        qna_pair = "<SOQ> " + query + "\n"
        qna_pair += "<SOC> "
        for passage in passages:
            qna_pair += re.sub(r"\s", " ", passage) + "\n"
        qna_pair += "<SOA> "
        return qna_pair
        
    def generate_answer(self, question: str, context) -> str:
        model_input = self.build_model_input(question, context)
        encoded_prompt = self.tokenizer.encode(model_input, return_tensors="pt")
        if len(encoded_prompt[0]) > 974:
            return ''

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=len(encoded_prompt[0]) + 50,
            # do_sample=True,
            # early_stopping=True,
            # num_beams=4,
            # top_p=0.92,
            # temperature=0.9,
            # no_repeat_ngram_size=4,
        )
        generated_sequence = output_sequences[0].tolist()

        # Decode text
        text = self.tokenizer.decode(
            generated_sequence, clean_up_tokenization_spaces=True)

        # Cut the initial input
        text = text[len(self.tokenizer.decode(encoded_prompt[0],
                                        clean_up_tokenization_spaces=True)):]

        # Remove all text after the stop token
        text = text[: text.find(self.stop_token)]

        return text.strip()


class SquadAnswerer:
    """
    This class uses a squad model to answer questions.
    """

    def __init__(self, model_path, cache_dir=None):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def build_context(self, passages):
        result = ''
        for passage in passages:
            result += re.sub(r"\s", " ", passage) + "\n"
        return result
        
    def generate_answer(self, question: str, context) -> str:
        try:
            result = self.pipeline(question=question, context=self.build_context(context))
            return result['answer']
        except Exception:
            return ""

marker_dict = {
    'question': ['Frage', 'question', 'query'],
    'context': ['Kontext', 'context', 'passages']
}

class T5Answerer:
    """
    This class uses a T5 model to answer questions.
    """

    def __init__(self, model_path, cache_dir=None):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

        self.question_marker = os.environ['T5_QUESTION_MARKER']
        self.context_marker = os.environ['T5_CONTEXT_MARKER']

        if not self.question_marker in marker_dict['question'] or not self.context_marker in marker_dict['context']:
            raise Exception("T5 markers aren't valid")

    def build_model_input(self, query, passages):
        qna_pair = self.question_marker + ": " + query + "\n"
        qna_pair += self.context_marker + ": "
        for passage in passages:
            qna_pair += re.sub(r"\s", " ", passage) + "\n"
        return qna_pair
        
    def generate_answer(self, question: str, context) -> str:
        model_input = self.build_model_input(question, context)
        encoded_prompt = self.tokenizer.encode(model_input, return_tensors="pt")
        if len(encoded_prompt[0]) > 2048:
            return ''

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=len(encoded_prompt[0]),
            # do_sample=True,
            # early_stopping=True,
            # num_beams=4,
            # top_p=0.92,
            # temperature=0.9,
            # no_repeat_ngram_size=4,
        )

        # Decode text
        return self.tokenizer.decode(output_sequences[0]).strip()


model_dict = {
    "gpt2": GPT2Answerer,
    "squad": SquadAnswerer,
    "t5": T5Answerer
}

def find_answerer(model_path):
    """
    returns the matching answerer for the given model.
    """
    for key in model_dict:
        if key in model_path.lower():
            return model_dict[key]
    return None
