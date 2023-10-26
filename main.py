import os
import bisect
from collections import defaultdict
from typing import List

import numpy as np
import torch

import nltk
import tiktoken
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tokenizer_path = "bert-base-uncased"
model_path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_position_embeddings = 512
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

def get_ppl(
    text: str,
    granularity: str = "sentence",
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    return_kv=False,
    end=None,
    condition_mode: str = "none",
    condition_pos_id: int = 0,
):
    if input_ids is None:
        tokenized_text = tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)

    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]
    else:
        past_length = 0
    
    if end is None:
        end = input_ids.shape[1]
    
    end = min(end, past_length + max_position_embeddings)

    with torch.no_grad():
        response = model(
            input_ids[:, past_length:end], #bxlen
            attention_mask=attention_mask[:, :end],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = response.past_key_values

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = response.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., past_length + 1 : end].contiguous()
    # Flatten the tokens
    active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
    active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
    active_labels = shift_labels.view(-1)[active]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(active_logits, active_labels)
    
    if condition_mode == "before":
        loss = loss[:condition_pos_id]
    elif condition_mode == "after":
        loss = loss[condition_pos_id:]
    res = loss.mean() if granularity == "sentence" else loss
    return (res, past_key_values) if return_kv else res


if __name__ == "__main__":
    # ppl test
    text = """That's the fun question. \
        If they're providing the wrong information then who knows! \
            Usually it's pretty obvious when you're exceeding the context because the results from the model turn to nonsense. \
                If the settings the people who made the model provide aren't actually accurate, doing a test to find the practical limit to the model's context is probably the only real way to know. That's out of scope for a simple tool like convert.py though."""
    res = get_ppl(text,
            granularity="token",
            )
    
    print(res)

