import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def parse_multi_intent_output(output_text, original_query, query_id=0):
    segments = output_text.strip().split(" | ")
    sub_queries = []
    intent_to_meta = {
        "exam_time": {"time_request": {}},
        "exam_location": {"location_request": {}},
        "assignment_deadline": {"time_request": {}},
        "submission_platform": {"location_request": {}},
        "lecture_notes": {"location_request": {}},
        "office_hours": {"time_request": {}},
        "resit_exam": {"time_request": {}},
        "group_project_policy": {"location_request": {}},
    }

    for i, segment in enumerate(segments):
        if "]" in segment:
            intent = segment.split("]")[0][1:].strip()
            text = segment.split("]")[1].strip()
        else:
            intent = "unknown"
            text = segment.strip()

        sub_queries.append({
            "id": i,
            "text": text,
            "matched": False,
            "match_result": None,
            "meta": intent_to_meta.get(intent, {})
        })

    return {
        "query_id": query_id,
        "original_query": original_query,
        "sub_queries": sub_queries
    }


def prompt_analyze(input_text, model, tokenizer, max_length=128):
    """
    Tokenize input, generate output using the model, and parse structured result.

    Args:
        input_text (str): The raw input query.
        model: A HuggingFace-compatible Seq2Seq model.
        tokenizer: The tokenizer associated with the model.
        max_length (int): The max length for generation.

    Returns:
        dict: Parsed multi-intent structured result.
    """
    # Tokenize and generate
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=max_length)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse structured output
    query_store = parse_multi_intent_output(decoded_output, original_query=input_text)
    prompts = [q['text'] for q in query_store['sub_queries']]
    return query_store,prompts
