import os
import gc
import json
import logging
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List
from peft import LoraConfig, TaskType, get_peft_model

from utils import get_model, evaluate, check_current_result, get_dataset
from prompt_template import get_prompt, get_prompt_with_qa
import prompt_template


def get_argument_parser():
    parser = argparse.ArgumentParser()
    # RUG
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", default=20, type=int)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample", type=int)  # -1:
    parser.add_argument("--task_name", type=str, choices=["icl", "ip"])
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--with_psg", action="store_true")
    parser.add_argument("--qa_generator", type=str, default="gpt")

    parser.add_argument("--method", type=str, default='direct')
    parser.add_argument("--qa_count", type=int, default=-1)

    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=3000)

    # Lora
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.01)
    return parser


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, max_length):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + \
                [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            })
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples],
                ['input_ids', 'labels', 'attention_mask'])
        )
        return {
            'input_ids': torch.tensor(input_ids).to(self.device),
            'labels': torch.tensor(labels).to(self.device),
            'attention_mask': torch.tensor(attention_mask).to(self.device),
        }


def predict(model, tokenizer, generation_config, question, passages, with_psg, with_cot, generate_qa=None):
    model.eval()
    if generate_qa is not None:
        input_ids = get_prompt_with_qa(
            tokenizer,
            question,
            passages,
            generate_qa,
            with_cot=with_cot
        )
    else:
        input_ids = get_prompt(
            tokenizer,
            question,
            passages=passages if with_psg else None,
            with_cot=with_cot)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(input_ids, **generation_config)
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    task_cfg = vars(args)

    if args.task_name == "icl":
        if args.method == "with_qa":
            if args.with_psg == False:
                raise ValueError("method == \"with_qa\" but without passage")
            else:
                full_task_name = "icl_with_psg_and_qa"
        else:
            full_task_name = "icl_with_psg" if args.with_psg else "wo_rag"
    else:
        full_task_name = "ip_with_psg" if args.with_psg else "ip_wo_psg"
    output_dir = os.path.join(args.output_dir,
                              args.model_name,
                              args.dataset,
                              'cot' if args.with_cot else 'direct',
                              full_task_name)
    same_task_path, cur_task_id = check_current_result(
        output_dir, task_cfg.copy())
    if same_task_path is None:
        os.makedirs(os.path.join(output_dir, cur_task_id), exist_ok=True)
        output_path = os.path.join(output_dir, cur_task_id)
        with open(os.path.join(output_path, 'config.json'), 'w') as fout:
            json.dump(task_cfg, fout, indent=4)
    else:
        output_path = same_task_path

    dataset = get_dataset(args.dataset, args.data_path, args.qa_generator)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    if "data_type" in task_cfg and args.data_type is not None:
        if args.data_type != 'full' and args.data_type is not None and args.data_type not in dataset.type_list:
            raise ValueError(f'unknown data type {args.data_type}')
        if isinstance(args.data_type, list):
            type_list = args.data_type
        else:
            type_list = [args.data_type]
    else:
        type_list = dataset.type_list

    if args.task_name == 'ip':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    for data_type in type_list:
        fulldata = dataset.qa_data[data_type]
        output_file = os.path.join(output_path, data_type + '.json')
        if os.path.exists(output_file):
            with open(output_file, "r") as fin:
                ret = json.load(fin)
                start_from = len(ret)
        else:
            start_from = 0
            ret = []
        if args.sample == -1:
            end_to = len(fulldata)
        else:
            end_to = args.sample

        for data in tqdm(fulldata[start_from:end_to]):

            test_id = len(ret)
            question = data['question']
            passages = data['passage']
            answer = data['answer']
            generate_qa = data['generate_qa']
            if args.qa_count != -1:
                generate_qa = generate_qa[:args.qa_count]

            if args.task_name == 'ip':
                # in parameter
                prompt_ids = []
                if args.method == 'direct':
                    for val in generate_qa:
                        psg = val['passage']
                        qpa_cnt = (len(val['qa']) + 1) // 2
                        for qid, qa in enumerate(val['qa']):
                            prompt_ids.append(get_prompt(
                                tokenizer,
                                qa['question'],
                                passages=[psg] if qid < qpa_cnt else None,
                                answer=qa['answer'] if not args.with_cot else qa['full_answer'],
                                with_cot=args.with_cot))
                elif args.method == "wo_qa":
                    prompt_ids.append(get_prompt(
                        tokenizer,
                        question,
                        passages=passages,
                        with_cot=args.with_cot,
                    ))

                train_data = TrainingData(
                    prompt_ids, tokenizer, args.block_size)
                device = model.device
                train_dataloader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=args.per_device_train_batch_size,
                    collate_fn=TrainingDataCollator(tokenizer, device),
                    shuffle=False,
                )

                model = get_peft_model(model, peft_config)
                model.is_parallelizable = True
                model.model_parallel = True
                model_parameters = filter(
                    lambda p: p.requires_grad, model.parameters())
                optimizer = torch.optim.AdamW(
                    model_parameters, lr=args.learning_rate)
                for epoch in range(args.num_train_epochs):
                    for step, batch in enumerate(train_dataloader):
                        # print(batch)
                        optimizer.zero_grad()
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        if args.logging_steps != -1 and step % args.logging_steps == 0:
                            print(
                                f"Epoch {epoch}, Step {step}, Loss {loss}, Learning rate {optimizer.param_groups[0]['lr']}")

            text = predict(model, tokenizer, generation_config,
                           question, passages,
                           with_psg=args.with_psg,
                           with_cot=args.with_cot,
                           generate_qa=generate_qa if args.task_name == "icl" and args.with_psg and args.method == "with_qa" else None)

            pred = {
                'test_id': test_id,
                'question': question,
                'predict': text,
                'answer': answer,
            }
            pred.update(evaluate(text, answer, args.with_cot))
            ret.append(pred)

            if args.task_name == 'ip':
                model.unload()
                torch.cuda.empty_cache()
                gc.collect()

            with open(output_file, 'w') as f:
                json.dump(ret, f, indent=4)

        with open(output_file, 'w') as f:
            json.dump(ret, f, indent=4)

        print(f"##### Evaluating #####")
        print(task_cfg)
        print(output_file)
        metrics = ['em', 'f1', 'prec', 'recall']
        for met in metrics:
            acc = sum(float(d[met]) for d in ret)
            acc /= len(ret)
            print(f"{met}\t{acc}")


if __name__ == '__main__':
    main()
