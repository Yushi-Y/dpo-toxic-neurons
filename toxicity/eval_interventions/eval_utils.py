"""
Utility functions to save/load models, data, etc.
"""
import json
import torch
from tabulate import tabulate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
# from transformer_lens import HookedTransformer


def tokenize(tokenizer, data, config):
    """
    Tokenize data.
    """
     # Ensure padding token is set before tokenization
    max_prompt_size = config.get("max_prompt_size")
    max_new_tokens = config.get("max_new_tokens")
    prompts = [x["prompt"] for x in data]

    if max_prompt_size is not None:
        tokenized = tokenizer(
            prompts,
            max_length=max_prompt_size,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    elif max_prompt_size is None and len(prompts) == 1:
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
        )
    else:
        raise RuntimeError("Unexpected data tokenization specification.")

    # Extract input_ids and attention mask
    prompt_input_ids = tokenized["input_ids"]
    prompt_attention_mask = tokenized["attention_mask"]

    gold = None
    gold_input_ids = None
    gold_attention_mask = None
    if all("gold" in x for x in data):
        gold = [x["gold"] for x in data]
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        gold_tokenized = tokenizer(
            gold,
            max_length=max_prompt_size + max_new_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        tokenizer.padding_side = orig_padding_side

        gold_input_ids = gold_tokenized["input_ids"]
        gold_attention_mask = gold_tokenized["attention_mask"]

    return {
        "prompts": prompts,
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "gold": gold,
        "gold_input_ids": gold_input_ids,
        "gold_attention_mask": gold_attention_mask,
    }



def load_model(config):
    """
    Load model, tokenizer and distribute across multiple GPUs if available.
    """
    assert "model_or_path" in config
    assert "tokenizer" in config

    tokenizer_name = config["tokenizer"]
    model_name = config["model_or_path"]
    state_dict_path = config.get("state_dict_path")
    state_dict = None

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)["state"]

    # Load model config
    # model_config = AutoConfig.from_pretrained(model_name)

    # # Detect if using a Gemma model
    # is_gemma = "gemma" in model_name.lower()

    # # Load model with appropriate attention handling
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     state_dict=state_dict,
    #     torch_dtype=torch.float32,
    #     attn_implementation="eager" if is_gemma else "sdpa"
    # ).to(config["device"])

    # # Load tokenizer
    # if tokenizer_name.startswith("gpt2"):
    #     tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    #     tokenizer.padding_side = "left"
    #     tokenizer.pad_token = tokenizer.eos_token  # GPT-2 lacks a native pad token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    #     tokenizer.padding_side = "left"

    # return model, tokenizer


    model = AutoModelForCausalLM.from_pretrained(
            model_name, state_dict=state_dict #, attn_implementation="eager"
            ).to(config["device"])

    # Distribute model across multiple GPUs if available 
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)

    # Load tokenizer
    if tokenizer_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    return model, tokenizer





def load_data(data_config):
    """
    Load data.
    NOTE: Expects a .jsonl file.
    """
    datapath = data_config["datapath"]

    if datapath.endswith(".jsonl"):
        with open(datapath, "r") as file_p:
            data = file_p.readlines()

        data = [json.loads(x.strip()) for x in data]
        return data

    assert "dataname" in data_config
    assert "split" in data_config
    data = load_dataset(
        datapath, data_config["dataname"], split=data_config["split"]
    )
    return [{"prompt": "\n\n".join(data["text"])}]



def pretty_print_results(results):
    """
    Pretty-print results.
    """
    metrics = None
    reformatted = []
    for intervene_method, _results in results.items():
        if metrics is None:
            metrics = list(_results.keys())

        reformatted.append([intervene_method] + [_results[k] for k in metrics])
    tabulated = tabulate(reformatted, headers=metrics, tablefmt="orgtbl")
    print(tabulated)



def get_intervene_name(config):
    """
    Construct a name for intervention config.
    """
    name = config["method"]
    if "params" in config:
        params = config["params"]
        if "type" in params:
            name += f"_{params['type']}"
        if "scale" in params:
            name += f"_scale:{params['scale']}"
        if "subtract_from" in params:
            name += f"_subtract_from:{params['subtract_from']}"
    return name
