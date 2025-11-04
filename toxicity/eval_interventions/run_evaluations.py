"""
Evaluation Module for interventions
"""
import sys
sys.path.append('/data/kebl6672/dpo-toxic-general/')

from typing import Dict

import os
import copy
import torch

# os.chdir('/code/dpo_toxic')

device = "cuda"

from toxicity.eval_interventions.eval_utils import ( 
    load_model,
    load_data,
    tokenize,
    get_intervene_name,
    pretty_print_results,
)
from toxicity.eval_interventions.generate_funcs import (
    generate_default,
    get_prompts,
    get_gold,
)
from toxicity.eval_interventions.metric_funcs import (
    run_f1,
    run_perplexity,
    run_perspective_api,
    run_n_gram_repetition, 
    run_dummy,
    run_detoxify_toxicity,
)
from toxicity.eval_interventions.hook_utils import (
    dont_hook,
    hook_subtract,
    scale_top_key_vectors,
    scale_top_value_vectors,
    scale_top_key_vectors_with_positive_activations,
    scale_top_value_vectors_with_positive_activations,
    assign_activations_to_neurons_full,
    assign_activations_to_neurons_new
)
from constants import (
    ROOT_DIR,
    PROFANITY,
    SEXUALLY_EXPLICIT,
    IDENTITY_ATTACK,
    THREAT,
    INSULT,
    SEVERE_TOXICITY,
    TOXICITY,
    PERSPECTIVE_API_ATTRIBUTES as ATTRIBUTES,
)
from utils import verbose_print, VERBOSE

DATA_DIR = os.path.join(ROOT_DIR, "data/intervene_data")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")


GENERATE_FUNCS = {
    "get_prompts": get_prompts,
    "get_gold": get_gold,
}
METRIC_FUNCS = {
    "f1": run_f1,
    "perplexity": run_perplexity,
    "dummy": run_dummy,
    "perspective_api": run_perspective_api,
    "n_gram_repetition": run_n_gram_repetition, 
    "detoxify": run_detoxify_toxicity,
}
HOOK_FUNCS = {
    "subtraction": hook_subtract,
    "scale_key_vector": scale_top_key_vectors,
    "scale_value_vector": scale_top_value_vectors,
    "scale_key_vector_with_positive_activation": scale_top_key_vectors_with_positive_activations,
    "scale_value_vector_with_positive_activation": scale_top_value_vectors_with_positive_activations,
    "assign_activations_to_neurons_full": assign_activations_to_neurons_full,
    "assign_activations_to_neurons_general": assign_activations_to_neurons_new
}
UNHOOK_FUNCS = {}


def generate(model, data, intervene_config):
    """
    Test intervention on a specific metric.
    """
    return GENERATE_FUNCS.get(intervene_config["method"], generate_default)(
        model, data, intervene_config["params"]
    )


def run_metric(
    metric_type,
    model,
    data_obj,
    intervene_results: Dict[str, torch.LongTensor],
    config,
):
    """
    Calculate specific metric.

    :intervene_results: Mapping from intervention specification to a tensor
        of shape [data_size, max_prompt_len + max_new_tokens]
    """
    return METRIC_FUNCS[metric_type](
        model,
        data_obj,
        intervene_results,
        config,
    )


def hook_model(model, config):
    """
    Hook model.
    """
    hook = HOOK_FUNCS.get(config["method"], dont_hook)(model, config["params"])
    
    # Ensure the hook(s) are always returned as a list, and make sure they're not tuples
    if isinstance(hook, tuple):
        hook = list(hook)  # Convert tuple to a list if needed
    elif not isinstance(hook, list):
        hook = [hook]
    
    return model, hook  # Return both the model and the hook(s)




def unhook_model(model, hooks):
    """
    Remove hooks in the model. Ensure 'hooks' is iterable to avoid errors.
    """
    # Ensure 'hooks' is a list or tuple to safely iterate over it
    if not isinstance(hooks, (list, tuple)):
        hooks = [hooks]
    
    # Iterate over the hooks and remove each one
    for hook in hooks:
        if hasattr(hook, "remove"):
            hook.remove()  # Safely remove only if the hook has a 'remove' method
        else:
            print(f"Warning: Hook {hook} does not have a remove method.")



def _eval_intervene(
    model, tokenizer, model_config, intervene_config, metric_configs
):
    """
    Evaluation intervention on set of metrics.
    """
    assert "method" in intervene_config
    intervene_config["params"]["device"] = model_config["device"]

    results = {}
    for _metric_conf in metric_configs:
        metric_type = _metric_conf["metric"]
        intervene_config["params"]["max_new_tokens"] = None

        verbose_print(f"Evaluating {metric_type}")
        data = _metric_conf["tokenized"]

        intervene_config["params"]["hook_timesteps"] = -1
        if metric_type == "perplexity":
            intervene_config["params"]["hook_timesteps"] = 0

        _, hooks = hook_model(model, intervene_config)

        generations = {}
        do_generate = _metric_conf["generate"]
        if do_generate:

            intervene_config["params"]["max_new_tokens"] = _metric_conf[
                "max_new_tokens"
            ]
            intervene_config["params"]["batch_size"] = model_config[
                "batch_size"
            ]
            generations = generate(model, data, intervene_config)
            for gen in generations["pred_text"][:30]:
                verbose_print(gen)
                # print(gen)

        results[metric_type] = run_metric(
            metric_type,
            model,
            data,
            generations,
            _metric_conf.get("params"),
        )
        # unhook_model(model, hooks)
    return results


def unroll_intervene(configs):
    """
    Unroll any nested configurations.
    """
    unrolled = []
    for _config in configs:
        method = _config["method"]
        if method != "subtraction":
            unrolled.append(_config)
            continue

        params = _config["params"]
        scales = params.pop("scales", [])
        if len(scales) < 1:
            raise RuntimeError("Missing scale value?")

        subtract_sets = params.pop("subtract_from", [])
        if len(subtract_sets) < 1:
            raise RuntimeError("Missing subtract_from value?")

        for scale in scales:
            for subtract_set in subtract_sets:
                config_copy = copy.deepcopy(_config)
                config_copy["params"]["scale"] = scale
                config_copy["params"]["subtract_from"] = subtract_set
                unrolled.append(config_copy)

    return unrolled


def tokenize_data(tokenizer, config):
    """
    Tokenize all data beforehand.
    """
    metric_configs = config["metrics"]

    tokenized_data = {}
    for _metric_conf in metric_configs:
        datapath = _metric_conf["datapath"]
        if datapath in tokenized_data:
            _metric_conf["tokenized"] = tokenized_data[datapath]
            continue

        data = load_data(_metric_conf)
        tokenized_data[datapath] = tokenize(tokenizer, data, _metric_conf)
        _metric_conf["tokenized"] = tokenized_data[datapath]


def run_eval(config):
    """
    Run eval!
    """
    model_config = config["model"]
    metric_configs = config["metrics"]
    interventions = config["interventions"]

    assert len(metric_configs) == len(
        list(set([x["metric"] for x in metric_configs]))
    ), "Mismatch -- you likely specified the same metric twice!"

    model, tokenizer = load_model(model_config)
    model.tokenizer = tokenizer

    # Set padding side for LLaMA model (decoder-only models)
    tokenizer.padding_side = "left"

    # Ensure tokenizer has a padding token set before proceeding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize all data beforehand.
    for _metric_conf in metric_configs:
        if "params" not in _metric_conf:
            _metric_conf["params"] = {}
        _metric_conf["params"]["pad_token_id"] = tokenizer.pad_token_id
        _metric_conf["params"]["batch_size"] = model_config["batch_size"]
        _metric_conf["params"]["device"] = model_config["device"]

    tokenize_data(tokenizer, config)

    interventions = unroll_intervene(interventions)
    results = {}
    for intervene_config in interventions:

        intervene_name = get_intervene_name(intervene_config)
        verbose_print(f"  Evaluating intervention {intervene_name}")

        results[intervene_name] = _eval_intervene(
            model, tokenizer, model_config, intervene_config, metric_configs
        )
        pretty_print_results(results)
    return results


def main():
    """ Driver """
    verbose_mode = VERBOSE  
    config = {
        "model": {
            "model_or_path": "gpt2-medium", #"gpt2-medium", # "meta-llama/Llama-3.1-8B", #"meta-llama/Llama-3.1-8B", # "gpt2-medium", "google/gemma-2-2b", # "mistralai/Mistral-7B-v0.1"
            # "state_dict_path": os.path.join(CKPT_DIR, "gemma2_2b_dpo_0.05.pt"), # Use the DPO model # dpo.pt #mistral_dpo.pt
            "tokenizer": "gpt2-medium", # "meta-llama/Llama-3.1-8B", # "mistralai/Mistral-7B-v0.1", # "meta-llama/Llama-3.1-8B", # "meta-llama/Llama-2-7b-hf", #"google/gemma-2-2b-it", #"mistralai/Mistral-7B-v0.1",#"google/gemma-2-2b", #"meta-llama/Meta-Llama-3-8B", # gpt2-medium
            "batch_size": 512,
            "device": device if torch.cuda.is_available() else "cpu",
        },
        "metrics": [
            # {
            #     "datapath": os.path.join(DATA_DIR, "challenge_prompts_dev.jsonl") if verbose_mode else os.path.join(DATA_DIR, "challenge_prompts.jsonl"),
            #     "metric": "perspective_api",
            #     "max_prompt_size": 32,
            #     "max_new_tokens": 20,
            #     "generate": True,
            #     "params": {"attributes": ATTRIBUTES, "include_prompt": False},
            # },
            {
                "datapath": os.path.join(DATA_DIR, "challenge_prompts_dev.jsonl") if verbose_mode else os.path.join(DATA_DIR, "challenge_prompts.jsonl"),
                "metric": "detoxify",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True,
                "params": {"include_prompt": False},  
            },
            {
                "datapath": "wikitext",
                "dataname": "wikitext-2-raw-v1",
                "split": "test",
                "metric": "perplexity",
                "generate": False,
            },
            {
                "datapath": os.path.join(DATA_DIR, "wiki_samples_dev.jsonl") if verbose_mode else os.path.join(DATA_DIR, "wiki_samples.jsonl"),
                "metric": "f1",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True,
            },
            {
                "datapath": os.path.join(DATA_DIR, "wiki_samples_dev.jsonl") if verbose_mode else os.path.join(DATA_DIR, "wiki_samples.jsonl"),
                "metric": "n_gram_repetition",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True, # True # False
                "params": {"include_prompt": False}, 
            },
        ],
        "interventions": [
            # {"method": "noop", "params": {}},
            # {
            #     "method": "subtraction",
            #     "params": {
            #         "type": "mlp_w_out",
            #         "idx": 770,
            #         "layer": 19,
            #         "subtract_from": [[23]],
            #         "scales": [20],
            #     }
            # },
            # {
            #     "method": "subtraction",
            #     "params": {
            #         "type": "toxic_probe",
            #         "scales": [1],
            #         "subtract_from": [[25]], # 23 (gpt2) # 31 (llama3, mistral) # 41 (gemma2 9b) # 25 (gemma2 2b)
            #         "datapath": os.path.join(CKPT_DIR, "gemma2_2b_probe.pt"),
            #     }
            # },
            # {
            #     "method": "subtraction",
            #     "params": {
            #         "type": "svd",
            #         "idx": 0,
            #         "scales": [100],
            #         "subtract_from": [[23]],
            #         "topk_sorted_score": 512,
            #         "datapath": os.path.join(CKPT_DIR, "probe.pt"),
            #     }
            # },
            # {
            #      "method": "scale_key_vector", 
            #      "params": {
            #          "probe_vector_path": os.path.join(CKPT_DIR, "probe.pt"),
            #          "topk_sorted_score": 7,
            #          "scale_factor": 10
            #     }
            # },
            # {
            #      "method": "scale_value_vector", 
            #      "params": {
            #          "probe_vector_path": os.path.join(CKPT_DIR, "gpt2_lee_probe.pt"),
            #          "topk_sorted_score": 60,
            #          "scale_factor": 0
            #     }
            # },
            # {
            #      "method": "scale_value_vector", 
            #      "params": {
            #          "probe_vector_path": os.path.join(CKPT_DIR, "gpt2_lee_probe.pt"),
            #          "topk_sorted_score": 128,
            #          "scale_factor": 0
            #     }
            # }
            # {
            #      "method": "scale_key_vector_with_positive_activation", 
            #      "params": {
            #          "topk_sorted_score": 3000,
            #          "scale_factor": 0,
            #          "toxic_positive_acts_index_csv_path": "/code/dpo_toxic/toxic_positive_acts_idxs.csv"
            #     }
            # }
            # {
            #      "method": "scale_value_vector_with_positive_activation", 
            #      "params": {
            #          "topk_sorted_score": 36,
            #          "scale_factor": 0,
            #          "toxic_positive_acts_index_csv_path": "/data/kebl6672/dpo-toxic-neuron/toxic_positive_acts_idxs.csv"
            #     }
            # }
            # {
            #      "method": "revert_activations", 
            #      "params": {
            #         "probe_vector_path": os.path.join(CKPT_DIR, "probe.pt"),
            #          "topk_sorted_score": 1,
            #          "modification_value": [1]
            #          }
            # }
            # {
            #      "method": "zero_out_activation_at_neuron", 
            #      "params": {
            #         "layer_index": 19,
            #         "neuron_index": 770
            #          }
            # }
        #     {
        #          "method": "print_and_return_activation_values", 
        #          "params": {
        #             "layer_index": 19,
        #             "neuron_index": 770
        #              }
        #     }
            {
                 "method": "assign_activations_to_neurons_full", 
                 "params": {
                    "neuron_configs_path": '/data/kebl6672/dpo-toxic-general/toxicity/activation_analysis/gpt2_0.99_1.01_two_0.55_cossim_embed_dpo.csv'
                }
            }
        ],
    }
    results = run_eval(config)
    print("Final Results:")
    pretty_print_results(results)


if __name__ == "__main__":
    main()


