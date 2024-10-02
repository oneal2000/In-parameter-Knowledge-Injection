import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import HotpotQA, WikiMultiHopQA, TriviaQA, BaseDataset


def get_model_path(model_name):
    if model_name == "llama3-8b-instruct":
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return "Qwen/Qwen2.5-1.5B-Instruct"
    elif model_name == "llama3.2-1b-instruct":
        return "meta-llama/Llama-3.2-1B-Instruct"
    else:
        raise ValueError(f"unknown model {model_name}.")


def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    generation_config = dict(
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    return model, tokenizer, generation_config


def get_dataset(dataset_name, qa_generator, data_path):
    if dataset_name == "2wikimultihopqa":
        return WikiMultiHopQA(data_path, qa_generator)
    elif dataset_name == "triviaqa":
        return TriviaQA(data_path, qa_generator)
    else:
        raise ValueError(f"unknown dataset {dataset_name}.")


base_dataset_template = BaseDataset()


def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "the answer is" in pred:
            pred = pred[pred.find("the answer is") + len("the answer is"):]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()

    em = base_dataset_template.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = base_dataset_template.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def check_current_result(dirpath, cur_task_cfg):
    cur_task_cfg['sample'] = -1
    cur_task_cfg['data_type'] = None
    if "qa_count" not in cur_task_cfg:
        cur_task_cfg["qa_count"] = -1
    if os.path.exists(dirpath):
        for dirname in os.listdir(dirpath):
            with open(os.path.join(dirpath, dirname, 'config.json'), 'r') as fin:
                file_cfg = json.load(fin)
            file_cfg['sample'] = -1
            file_cfg['data_type'] = None

            if "qa_count" not in file_cfg:
                file_cfg["qa_count"] = -1
            if cur_task_cfg == file_cfg:
                return os.path.join(dirpath, dirname), None
        else:
            all_idx = [int(k) for k in os.listdir(dirpath)]
            for i in range(len(all_idx)+10):
                if i not in all_idx:
                    return None, str(i)
    else:
        return None, "0"
