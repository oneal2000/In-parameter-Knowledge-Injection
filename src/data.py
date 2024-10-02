import os
import json
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Callable, Tuple, Union, Callable


DATA_BASE_ROOT = "/liuzyai04/thuir/tyc/Dataset/"
QA_PATH = "generate_qa/"


class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(
            ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction)
                         == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(
            ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


class TriviaQA(BaseDataset):
    def __init__(self, data_path, qa_generator="gpt"):
        with open(data_path + 'rc/qa/verified-wikipedia-dev.json') as fin:
            dataset = json.load(fin)
            dataset = dataset['Data']
        self.test_data = []
        for did, data in enumerate(dataset):
            self.test_data.append({
                'qid': data['QuestionId'],
                'test_id': did,
                'question': data['Question'],
                'answer': data['Answer']['Aliases']
            })
        self.qa_data = {}
        self.type_list = ['small']
        for typ in self.type_list:
            with open(os.path.join(QA_PATH, "triviaqa", qa_generator, typ+".json"), "r") as fin:
                self.qa_data[typ] = json.load(fin)


class WikiMultiHopQA(BaseDataset):
    def __init__(self, data_path, qa_generator="gpt"):
        with open(data_path + 'dev.json', 'r') as fin:
            dataset = json.load(fin)
        with open(data_path + 'id_aliases.json', 'r') as fin:
            aliases = dict()
            for li in fin:
                t = json.loads(li)
                aliases[t['Q_id']] = t['aliases']
        self.test_data = []
        for did, data in enumerate(dataset):
            ans_id = data['answer_id']
            self.test_data.append({
                'qid': data['_id'],
                'test_id': did,
                'context': data['context'],
                'question': data['question'],
                'answer': aliases[ans_id] if ans_id else data['answer']
            })
        self.type_list = ["comparison", "bridge_comparison"]

        self.qa_data = {}
        for typ in self.type_list:
            if not os.path.exists(os.path.join(QA_PATH, "2wikimultihopqa", qa_generator, typ+".json")):
                self.qa_data[typ] = []
            else:
                with open(os.path.join(QA_PATH, "2wikimultihopqa", qa_generator, typ+".json"), "r") as fin:
                    self.qa_data[typ] = json.load(fin)
