import openai
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

openai.api_key = None

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

prompt_template = "I will provide a passage of text, and you need to generate six different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least 6 elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"


def generate(content):
    completion = openai.Completion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )
    return completion


def main():
    filepath = "path_to_2wikimultihopqa_dev_set"
    with open(os.path.join(filepath), "r") as fin:
        full_dataset = json.load(fin)

    type_list = list(set(d['type'] for d in full_dataset))

    for type_name in type_list:
        dataset = []
        for data in full_dataset:
            if data['type'] == type_name:
                dataset.append(data)

        output_file = f"../data/2wikimultihopqa/gpt/{type_name}.json"
        if os.path.exists(output_file):
            with open(output_file, "r") as fin:
                new_dataset = json.load(fin)
        else:
            new_dataset = []
        start_from = len(new_dataset)

        for data in tqdm(dataset[start_from:]):
            context = {}
            for name, content in data['context']:
                context[name] = " ".join(content)
            data['passage'] = [context[fact[0]]
                               for fact in data['supporting_facts']]

            data["generate_qa"] = []
            for pid, psg in enumerate(data['passage']):
                prompt = prompt_template.format(passage=psg)
                while True:
                    completion = generate(prompt)
                    output = completion.choices[0].message.content
                    try:
                        qa = json.loads(output)
                    except:
                        qa = []
                        continue
                    if len(qa) < 6:
                        continue
                    data['generate_qa'].append({
                        'passage_id': pid,
                        'passage': psg,
                        "gpt_origin_output": output,
                        "qa": qa,
                    })
                    break
            new_dataset.append(data)
        with open(output_file, "w") as fout:
            json.dump(new_dataset, fout, indent=4)


if __name__ == "__main__":
    main()
