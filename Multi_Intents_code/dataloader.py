import json
from datasets import Dataset

def load_dataset(json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        input_text = item["text"]
        output_lines = [f"[{q['intent']}] {q['text']}" for q in item["sub_queries"]]
        output_text = " | ".join(output_lines)
        data.append({"input": input_text, "output": output_text})
    return Dataset.from_list(data)

