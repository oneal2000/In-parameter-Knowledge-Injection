# In-parameter-Knowledge-Injection

## Overview




## Install environment

```
conda create -n in_para python=3.10.4
conda activate in_para
pip install torch==1.13.1
pip install -r requirements.txt
```

## Dataset

For 2WikiMultihopQA: 

Download the [2WikiMultihop](https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file) dataset from its repository https://www.dropbox.com/scl/fi/aasqsj45yokx71pnm8ctr/data_ids.zip?dl=0&e=1&file_subpath=%2Fdata_ids&rlkey=72n2p6jywhfmm6kdeuzz8c55u. Unzip it and move the folder to `dataset/2wikimultihopqa/`.

For TriviaQA:

```
mkdir -p dataset/triviaqa
wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
tar -xvf triviaqa-rc.tar.gz -C dataset/triviaqa/
```

## Generate QA pairs

The QA pairs generated for each passage can be found in the `data` folder.

You can also regenerate QA pairs using `src/get_qa.py. The file includes code for generating QA pairs with the 2WikiMultihopQA dataset, along with the prompts we used for generation.

## Run

Execute the code using `src/generate.py` with the following arguments:

### Argument Descriptions

| Argument                     | Type    | Description                                     |
|------------------------------|---------|-------------------------------------------------|
| `model_name`               | str     | Model name                            |
| `max_new_tokens`           | int     | Max new tokens to generate              |
| `data_path`                | str     | Path to the dataset, like `dataset/triviaqa`                   |
| `dataset`                  | str     | Dataset name (`triviaqa` or `2wikimultihopqa`)                          |
| `data_type`                | str     | Data type you want to use; if not specified, all data types will be used by default                                    |
| `output_dir`               | str     | Directory for output                  |
| `sample`                   | int     | Number of samples (-1 for all)         |
| `task_name`                | str     | Task name (`icl` or `ip`)                         |
| `with_cot`                 | flag    | Include if using chain of thought               |
| `with_psg`                 | flag    | Include if using passages                       |
| `qa_generator`             | str     | QA generator model, like `gpt`                  |
| `method`                   | str     | Method used (default: `direct`)                 |
| `qa_count`                 | int     | The number of QAs to augment generation                         |
| `per_device_train_batch_size` | int  | Training batch size per device    |
| `num_train_epochs`         | int     | Number of training epochs          |
| `learning_rate`            | float   | Learning rate                      |
| `logging_steps`            | int     | Logging intervals                  |
| `block_size`               | int     | Block size                      |
| `lora_rank`                | int     | LoRA rank                          |
| `lora_alpha`               | int     | LoRA alpha                         |
| `lora_dropout`             | float   | LoRA dropout rate                |

### Example Usage

```bash
python3 src/generate.py 
    --model_name llama3.2-1b-instruct 
    --max_new_tokens 20 
    --data_path dataset/2wikimultihopqa/ 
    --dataset 2wikimultihopqa 
    --data_type comparison 
    --output_dir output/ 
    --sample 300 
    --with_cot
    --qa_generator gpt 
    --method direct 
    --qa_count 3 
    --per_device_train_batch_size 1 
    --num_train_epochs 3 
    --learning_rate 0.0003
    --logging_steps -1 
    --block_size 3000
    --lora_rank 2
    --lora_alpha 32
    --lora_dropout 0.01
```

## Results

The results will be saved in the directory structure: 
```
output_dir/model_name/dataset_name/if_cot/method/idx/
```

The `idx` indicates the result of the nth run under the same task.

Inside this directory, you will find files named as `data_type.json`. Each file is in the following format:

```json
[
    {
        "test_id": 0,
        "question": "Where is Paris located?",
        "predict": "Paris is the capital of France. So the answer is France.",
        "answer": [
            "France", 
        ],
        "eval_predict": "France",
        "em": "1",
        "f1": "1.0",
        "prec": "1.0",
        "recall": "1.0"
    }
]
```