import os
import sys

os.chdir('/data/kebl6672/dpo-toxic-general/toxicity')
sys.path.append('/data/kebl6672/dpo-toxic-general/toxicity')

import json
import torch
import torch.nn.functional as F
# from transformer_lens import (
#     HookedTransformer,
# )
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
from fig_utils import load_model

device = torch.device("cuda") 
ROOT_DIR = '/data/kebl6672/dpo-toxic-general/ignore'

model_name = "gpt2-medium" # "google/gemma-2-2b" # "google/gemma-2-2b" # "mistralai/Mistral-7B-v0.1" # "meta-llama/Llama-3.1-8B" # "gpt2-medium" # "meta-llama/Llama-3.1-8B" # "google/gemma-2-2b", # "gpt2-medium", # "mistralai/Mistral-7B-v0.1",
# dpo_model_name = "gpt2_dpo.pt" # "gemma2_2b_dpo_0.05_final.pt" # "llama3_dpo_0.1_attn_final.pt" # "mistral_dpo_0.05_final.pt" # "gpt2_dpo.pt" # "llama3_dpo_2.pt"
probe_name = "gpt2_toxic_embed.pt" # "gemma2_2b_probe.pt" # "mistral_probe.pt" # "gpt2_probe.pt" # "llama3_probe.pt"
model_short_name = "gpt2" # "gemma2" "mistral" #"gpt2"
BATCH_SIZE = 512

## Load the tokenizer and model
config = {"model_or_path": model_name, "tokenizer": model_name, "device": "cuda"}
model, tokenizer = load_model(config)

# Load the DPO-ed model
# config_dpo = {
#     "model_or_path": model_name,
#     "tokenizer": model_name,
#     "device": "cuda",
#     "state_dict_path": os.path.join(ROOT_DIR, dpo_model_name),
# }
# dpo_model, tokenizer = load_model(config_dpo)

# Load the toxic vector
toxic_vector = torch.load(os.path.join(ROOT_DIR, probe_name)).to(device)  


# load evaluation data
DATA_DIR = '/data/kebl6672/dpo-toxic-neuron/data/intervene_data'

with open(
    os.path.join(DATA_DIR, "challenge_prompts.jsonl"), "r"
) as file_p:
    data = file_p.readlines()

prompts = [json.loads(x.strip())["prompt"] for x in data]

# Tokenizing the prompts correctly
tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
attention_mask = (tokenized_prompts != tokenizer.pad_token_id).long().to(device)  




def compute_neuron_toxic_projection(model, tokenized_prompts, toxic_vector, batch_size=BATCH_SIZE):
    """
    Computes neuron toxicity projections by extracting activations
    after non-linearity using hooks on c_proj (second MLP weight matrix).
    """
    model = model.module if isinstance(model, torch.nn.DataParallel) else model

    device = next(model.parameters()).device
    sample_size = tokenized_prompts.size(0)
    
    # Normalize toxic vector
    toxic_vector = toxic_vector.to(device, dtype=torch.float16).squeeze(0)
    toxic_norm = torch.norm(toxic_vector) # A scalar
    print(f"Toxic vector shape (d, ): {toxic_vector.shape}")

    # Store cumulative sums and counts
    intermediate_size = 4 * model.config.n_embd
    neuron_act_sums = defaultdict(lambda: torch.zeros(intermediate_size, dtype=torch.float16, device=device)) # m_i*v_i - (d)
    neuron_proj_sums = defaultdict(lambda: torch.zeros(intermediate_size, dtype=torch.float16, device=device)) # (m_i*v_i)*W - scalar
    neuron_counts = defaultdict(lambda: torch.zeros(intermediate_size, dtype=torch.int32, device=device))
    print(f"Dictionary size for each layer key (d_mlp): {intermediate_size}")

    # Store activations extracted from inputs to c_proj
    neuron_acts_storage = {}

    def hook_fn(module, input, output, layer_idx):
        neuron_acts_storage[layer_idx] = input[0].detach()  # Capture INPUT to c_proj

    # Register hooks - triggered in forward pass
    hooks = [
        model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(
            lambda module, input, output, l=layer_idx: hook_fn(module, input, output, l)
        )
        for layer_idx in range(len(model.transformer.h))
    ]

    print("Computing MLP neuron projections...")
    for idx in tqdm(range(0, sample_size, batch_size)):
        batch = tokenized_prompts[idx: idx + batch_size].to(device)
        batch_attention_mask = attention_mask[idx: idx + batch_size].to(device)

        with torch.inference_mode():
            generated_tokens = model.generate(batch, max_new_tokens=20, attention_mask=batch_attention_mask)  
            extended_batch = torch.cat([batch, generated_tokens[:, batch.shape[1]:]], dim=1)  # (B, T+20), append new tokens
            extended_attention_mask = torch.cat([batch_attention_mask, torch.ones_like(generated_tokens[:, batch.shape[1]:])], dim=1) 
            outputs = model(extended_batch, attention_mask=extended_attention_mask, output_hidden_states=True, return_dict=True) # Forward pass to capture activations

        torch.cuda.empty_cache()

        # Process activations only for 20 generated tokens
        for layer_idx, neuron_acts in neuron_acts_storage.items():
            print(f"Processing layer {layer_idx} activations with shape (B, T+20, d_mlp): {neuron_acts.shape}")
            value_vectors = model.transformer.h[layer_idx].mlp.c_proj.weight.to(device)  # (d_mlp, d)
            print(f"Value vectors shape at layer {layer_idx} (d_mlp, d): {value_vectors.shape}")
            
            # Compute scaling factor (d_mlp,)
            v = torch.matmul(value_vectors.to(toxic_vector.dtype), toxic_vector) / toxic_norm  # (d_mlp)
            print(f"Layer {layer_idx} value vector projection shape (d_mlp): {v.shape}")

            # Capturing the last 20 generated token activations
            neuron_acts_gen = neuron_acts[:, -20:, :]  # (B, 20, d_mlp)
            print(f"Extracting last 20 tokens (B, 20, d_mlp): {neuron_acts_gen.shape}")

            # Scale activations (B, 20, d_mlp)
            neuron_projections = neuron_acts_gen.clone() * v  # (B, 20, d_mlp) # Element-wise multiplication
            print(f"Final neuron projections shape (B, 20, d_mlp): {neuron_projections.shape}")

            # Accumulate sum of activations and projections and count for running mean per neuron
            neuron_act_sums[layer_idx] += neuron_acts_gen.sum(dim=(0, 1))  # Sum over (B, 20)
            neuron_proj_sums[layer_idx] += neuron_projections.sum(dim=(0, 1))  # Sum over (B, 20)
            neuron_counts[layer_idx] += neuron_projections.shape[0] * neuron_projections.shape[1]  # (B*20)
            print(f"Updated neuron sums for layer {layer_idx} with shape (d_mlp): {neuron_proj_sums[layer_idx].shape}") # (d_mlp)
            print(f"Updated neuron counts for layer {layer_idx} with shape (d_mlp): {neuron_counts[layer_idx].shape}") # (d_mlp)
            print(f"Neuron counts at each entry (B*20): {neuron_counts[layer_idx][3]}") # (B*20)

    for hook in hooks:
        hook.remove()

    # Compute final average neuron projections per neuron
    avg_neuron_projections = {
        (layer_idx, neuron_idx): (neuron_proj_sums[layer_idx][neuron_idx] / neuron_counts[layer_idx][neuron_idx]).cpu().item()
        for layer_idx in neuron_proj_sums
        for neuron_idx in range(neuron_proj_sums[layer_idx].shape[0])
    }

    # Compute final average neuron activations per neuron
    avg_neuron_activations = {
        (layer_idx, neuron_idx): (neuron_act_sums[layer_idx][neuron_idx] / neuron_counts[layer_idx][neuron_idx]).cpu().item()
        for layer_idx in neuron_act_sums
        for neuron_idx in range(neuron_act_sums[layer_idx].shape[0])
    }
    
    return avg_neuron_projections, avg_neuron_activations





def save_neuron_projections_to_csv(avg_neuron_projections, avg_neuron_activations, model_name):
    """Saves the neuron projection and activation data to a CSV file."""
    data = [
        {
            "layer_idx": layer_idx,
            "neuron_idx": neuron_idx,
            "projection_value": avg_neuron_projections.get((layer_idx, neuron_idx), None), 
            "activation_value": avg_neuron_activations.get((layer_idx, neuron_idx), None)
        }
        for (layer_idx, neuron_idx) in set(avg_neuron_projections.keys()).union(avg_neuron_activations.keys())
    ]

    df = pd.DataFrame(data, columns=["layer_idx", "neuron_idx", "projection_value", "activation_value"])
    
    filename = f"{model_name}_neuron_projections.csv"
    df.to_csv(filename, index=False)
    print(f"Neuron projections and activations saved to {filename}")




def compute_all_neuron_cossims(model, toxic_vector, model_name):
    """
    Computes the cosine similarity between each neuron's value vector (W_out rows) 
    and the toxic vector in a memory-efficient way using torch.nn.functional.cosine_similarity.
    """
    model = model.module if isinstance(model, torch.nn.DataParallel) else model

    device = next(model.parameters()).device
    toxic_vector = toxic_vector.to(device, dtype=torch.float16)  # (1,d)
    # print("toxic_vector (1,d):", toxic_vector.shape)  

    model_neuron_cossims = []

    for layer_idx, layer in enumerate(model.transformer.h):
        W_out = layer.mlp.c_proj.weight.to(torch.float16)  # (d_mlp, d)
        print(f"Layer {layer_idx}: W_out shape (d_mlp, d): {W_out.shape}")  

        # Compute cosine similarity directly
        cossims = F.cosine_similarity(W_out, toxic_vector, dim=1)  # (d_mlp)
        print(f"Layer {layer_idx}: cosine similarity shape (d_mlp): {cossims.shape}")

        model_neuron_cossims.extend([
            {"layer_idx": layer_idx, "neuron_idx": neuron_idx, "cosine_similarity": cossims[neuron_idx].item()}
            for neuron_idx in range(W_out.shape[0])
        ])

    df = pd.DataFrame(model_neuron_cossims)
    csv_filename = f"{model_name}_neuron_cossims.csv"
    df.to_csv(csv_filename, index=False)
    # print(f"Cosine similarities saved to {csv_filename}")

    return df





def main():
    """Main execution pipeline."""
    ### Compute and save neuron projections for the base model
    print("Processing pre-trained model...")
    # avg_neuron_projections, avg_neuron_activations = compute_neuron_toxic_projection(model, tokenized_prompts, toxic_vector)
    # save_neuron_projections_to_csv(avg_neuron_projections, avg_neuron_activations, model_short_name)
    compute_all_neuron_cossims(model, toxic_vector, model_short_name)

    ### Compute and save neuron projections for the DPO-trained model
    # print("Processing DPO model...")
    # avg_neuron_projections_dpo, avg_neuron_activations_dpo = compute_neuron_toxic_projection(dpo_model, tokenized_prompts, toxic_vector)
    # save_neuron_projections_to_csv(avg_neuron_projections_dpo, avg_neuron_activations_dpo, model_short_name + "_dpo")
    # compute_all_neuron_cossims(dpo_model, toxic_vector, model_short_name + "_dpo")


if __name__ == "__main__":
    main()