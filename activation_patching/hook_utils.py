"""
Utility functions for hooking.
"""
from functools import partial
import torch
import torch.nn.functional as F
import pandas as pd
import ast
from collections import defaultdict


def get_svd_u_vec(model, toxic_vector, topk_sorted_score, U_idx):
    """
    Get the svd U vector
    toxic_vector: toxic_vector [d_model]
    topk_sorted_score: (int) vectors we want to get
    U_idx: which u vec
    """
    scores = []
    for layer in range(model.config.n_layer):
        # mlp_outs = model.blocks[layer].mlp.W_out
        # [d_mlp, d_model]
        mlp_outs = model.transformer.h[layer].mlp.c_proj.weight
        cos_sims = F.cosine_similarity(
            mlp_outs, toxic_vector.unsqueeze(0), dim=1
        )
        _topk = cos_sims.topk(k=100)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_vecs = [
        # model.blocks[x[2]].mlp.W_out[x[1]]
        model.transformer.h[x[2]].mlp.c_proj.weight[x[1]]
        for x in sorted_scores[:topk_sorted_score]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = torch.stack(top_vecs)

    svd = torch.linalg.svd(_top_vecs.transpose(0, 1))
    svd_U = svd.U.transpose(0, 1)
    return svd_U[U_idx]


def get_intervene_vector(model, config):
    """
    Get vector according to specifications in :config:
    """

    def _get_mlp_w_out(_config):
        layer = _config["layer"]
        idx = _config["idx"]
        return model.transformer.h[layer].mlp.c_proj.weight[idx]

    def _get_mlp_w_in(_config):
        w_in_idx = _config["w_ins"][0]
        layer = w_in_idx[0]
        idx = w_in_idx[1]
        return model.transformer.h[layer].mlp.c_fc.weight[:, idx]

    def _get_toxic_probe(_config):
        return torch.load(_config["datapath"])

    def _get_svd(_config):
        topk_sorted_score = _config["topk_sorted_score"]
        u_idx = _config["idx"]
        toxic_vector = torch.load(_config["datapath"])
        return get_svd_u_vec(model, toxic_vector, topk_sorted_score, u_idx)

    def _get_random(_config):
        shape = model.transformer.h[0].mlp.c_proj.weight[0].shape
        device = model.device
        return torch.rand(shape).to(device)

    return {
        "mlp_w_out": _get_mlp_w_out,
        "mlp_w_in": _get_mlp_w_in,
        "toxic_probe": _get_toxic_probe,
        "svd": _get_svd,
        "random": _get_random,
    }[config["type"]](config)


### Old version
# def hook_subtract(model, config):
#     intervene_vector = get_intervene_vector(model, config)
#     scale = config["scale"]
#     subtract_from = config["subtract_from"]
#     hook_timesteps = config["hook_timesteps"]

#     def patch(vec, _scale):
#         def hook(module, input, output):

#             _vec = vec.unsqueeze(0).unsqueeze(0)
#             if hook_timesteps == -1:
#                 _vec = _vec.repeat(output.shape[0], 1, 1)
#             else:
#                 _vec = _vec.repeat(output.shape[0], output.shape[1], 1)
#             output[:, hook_timesteps:, :] = output[:, hook_timesteps:, :] - (
#                 _scale * _vec
#             )
#             return output

#         return hook

#     hooks = []
#     for layer in subtract_from:
#         # hook = model.transformer.h[layer].mlp.c_proj.register_forward_hook(
#         hook = model.transformer.h[layer].mlp.register_forward_hook(
#             patch(intervene_vector, scale)
#         )
#         hooks.append(hook)
#     return model, hooks


def hook_subtract(model, config):
    """
    Hooks into the MLP layers of different models (GPT-2, Mistral, Llama, Gemma) to subtract a specified intervention vector.

    Args:
        model: The loaded language model.
        config (dict): Configuration containing:
            - "scale": Scaling factor for the intervention.
            - "subtract_from": List of layer indices to apply the intervention.
            - "hook_timesteps": Timestep index to apply the intervention (-1 for all timesteps).

    Returns:
        model with forward hooks registered.
    """
    intervene_vector = get_intervene_vector(model, config)
    scale = config["scale"]
    subtract_from = config["subtract_from"]
    hook_timesteps = config["hook_timesteps"]

    def patch(vec, _scale):
        def hook(module, input, output):
            """
            Applies an intervention by subtracting `vec` scaled by `_scale` from the model's output.
            """
            try:
                device = output.device  
                _vec = vec.clone().to(device)  

                # print(f"[DEBUG] Output shape: {output.shape}")
                # print(f"[DEBUG] Vec shape before adjustment: {_vec.shape} (on device: {_vec.device})")

                # Ensure _vec is 3D: (1, 1, hidden_dim)
                _vec = _vec.squeeze()  
                if _vec.dim() == 1:  # (hidden_dim,) -> expand to (1, 1, hidden_dim)
                    _vec = _vec.unsqueeze(0).unsqueeze(0)

                # Match _vec dimensions to different models' output dimensions 
                if output.dim() == 2:  # (batch_size, hidden_dim)
                    _vec = _vec.squeeze(1)  # Remove sequence dimension
                    _vec = _vec.expand(output.shape[0], -1)  # Expand across batch

                elif output.dim() == 3:  # (batch_size, seq_len, hidden_dim)
                    _vec = _vec.expand(output.shape[0], output.shape[1], -1)  # Expand across batch & sequence

                elif output.dim() == 1:  # Edge case: single hidden state (hidden_dim,)
                    _vec = _vec.squeeze(0)  # Remove batch dimension

                else:
                    raise RuntimeError(f"Unexpected output shape: {output.shape}, Vec shape: {_vec.shape}")

                # print(f"[DEBUG] Scale factor: {_scale}, Vec shape after adjustment: {_vec.shape}, Vec norm: {vec.norm().item()}")

                output -= _scale * _vec 
                return output  

            except Exception as e:
                print(f"[ERROR] Hook function encountered an error: {e}")
                return output  # Ensure output is always returned

        print(f"[DEBUG] Hook function created successfully!")  
        return hook

    hooks = []
    for layer in subtract_from:
        try:
            hook_fn = patch(intervene_vector, scale) 

            if "gemma" in model.__class__.__name__.lower() or "llama" in model.__class__.__name__.lower():
                hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)

            elif "mistral" in model.__class__.__name__.lower():
                hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)

            elif "gpt2" in model.__class__.__name__.lower():
                hook = model.transformer.h[layer].mlp.register_forward_hook(hook_fn)

            else:
                raise ValueError(f"Unsupported model architecture: {type(model).__name__}")

            if hook is None:
                print(f"[ERROR] Hook registration failed for layer {layer}!")
            else:
                print(f"[DEBUG] Hook successfully registered for layer {layer}, Scale: {scale}") 

            hooks.append(hook)

        except Exception as e:
            print(f"[ERROR] Hook registration failed for layer {layer}: {e}")

    return model, hooks




def dont_hook(model, config):
    return model, []


def scale_top_key_vectors(model, config):
    """
    Scales the key vectors of the MLPs based on the cosine similarity of the value vectors to a given toxic vector.
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    toxic_vector = torch.load(config['probe_vector_path'])
    topk_sorted_score = config['topk_sorted_score']
    scale_factor = config['scale_factor']
    scores = []
    
    for layer in range(model.config.n_layer):
        value_vectors = model.transformer.h[layer].mlp.c_proj.weight
        cos_sims = F.cosine_similarity(value_vectors, toxic_vector.unsqueeze(0), dim=1)
        _topk = cos_sims.topk(k=1000)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    # print(sorted_scores[:20])

    top_key_vecs = [
        model.transformer.h[x[2]].mlp.c_fc.weight[:, x[1]]
        for x in sorted_scores[:topk_sorted_score]
    ]
    with torch.no_grad():
        for tensor in top_key_vecs:
            tensor *= scale_factor

    # For consistency, return model and an empty list of hooks
    return model, [] 



def scale_top_value_vectors(model, config):
    """
    Scales the value vectors of the MLPs based on the cosine similarity of the value vectors to a given toxic vector.
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    toxic_vector = torch.load(config['probe_vector_path'])  # Load the toxic vector
    topk_sorted_score = config['topk_sorted_score']  # Number of top vectors to scale
    scale_factor = config['scale_factor']  # Factor by which to scale the vectors
    scores = []
    
    for layer in range(model.config.n_layer):
        # Now, target the value vectors in the MLP
        value_vectors = model.transformer.h[layer].mlp.c_proj.weight
        
        # Compute cosine similarities between the value vectors and the toxic vector
        cos_sims = F.cosine_similarity(value_vectors, toxic_vector.unsqueeze(0), dim=1)
        
        # Get the top k most similar value vectors
        _topk = cos_sims.topk(k=1000)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    # Sort the scores in descending order based on cosine similarity
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    
    # Select the top `topk_sorted_score` value vectors and scale them
    top_value_vecs = [
        model.transformer.h[x[2]].mlp.c_proj.weight[x[1], :]
        for x in sorted_scores[:topk_sorted_score]
    ]
    
    # Scale the selected value vectors
    with torch.no_grad():
        for tensor in top_value_vecs:
            tensor *= scale_factor

    return model, [] 




def scale_top_key_vectors_with_positive_activations(model, config):
    """
    Scales the key vectors for neurons with positive activations before DPO based on the ranks of their cosine similarity with the toxic probe.
    
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    topk_sorted_score = config['topk_sorted_score']  
    scale_factor = config['scale_factor']  
    
    # Load the sorted scores from the CSV
    sorted_scores_df = pd.read_csv(config['toxic_positive_acts_index_csv_path'])
    
    # Select the top `topk_sorted_score` layer and neuron indices from the CSV
    top_layer_neuron_indices = sorted_scores_df.head(topk_sorted_score)
    
    # Scale the selected value vectors
    with torch.no_grad():
        for _, row in top_layer_neuron_indices.iterrows():
            layer_idx = row['layer_idx']
            neuron_idx = row['neuron_idx']
            tensor = model.transformer.h[layer_idx].mlp.c_fc.weight[:, neuron_idx]
            tensor *= scale_factor

    return model, [] 



def scale_top_value_vectors_with_positive_activations(model, config):
    """
    Scales the value vectors with positive activations before DPO based on the ranks of their cosine similarity with the toxic probe.
    
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    topk_sorted_score = config['topk_sorted_score']  
    scale_factor = config['scale_factor']  
    
    # Load the sorted scores from the CSV
    sorted_scores_df = pd.read_csv(config['toxic_positive_acts_index_csv_path'])
    
    # Select the top `topk_sorted_score` layer and neuron indices from the CSV
    top_layer_neuron_indices = sorted_scores_df.head(topk_sorted_score)
    
    # Scale the selected value vectors
    with torch.no_grad():
        for _, row in top_layer_neuron_indices.iterrows():
            layer_idx = row['layer_idx']
            neuron_idx = row['neuron_idx']
            tensor = model.transformer.h[layer_idx].mlp.c_proj.weight[neuron_idx, :]
            tensor *= scale_factor

    return model, [] 




# def assign_activations_to_neurons_gpt2(model, config):
#     """
#     Modify the activation coefficients for specific neurons in different layers.
#     Each entry in the config contains a tuple (layer_idx, neuron_idx, assigned_value).
#     """
#     neuron_configs = config['neuron_configs']  # List of (layer_idx, neuron_idx, assigned_value)
#     hook_timesteps = config["hook_timesteps"]

#     # Convert neuron configs to tensors for batch processing 
#     layer_idxs = torch.tensor([cfg[0] for cfg in neuron_configs], device="cuda")
#     neuron_idxs = torch.tensor([cfg[1] for cfg in neuron_configs], device="cuda")
#     assigned_values = torch.tensor([cfg[2] for cfg in neuron_configs], device="cuda")

#     def patch(layer_idx, neuron_idx_tensor, assigned_value_tensor):
#         def hook(module, input, output):
#             """
#             Forward hook to assign specific values to multiple neurons' activation coefficients in parallel.
#             """
#             with torch.no_grad():
#                 output[:, hook_timesteps, neuron_idx_tensor] = assigned_value_tensor
#                 print(f"Assigned values to neurons in layer {layer_idx}") # {neuron_idx_tensor.tolist()}

#             return output  # Return the modified pre-GELU activation

#         return hook

#     hooks = []

#     # Group by layer for parallel assignment within each layer
#     layer_groups = {}
#     for i, layer_idx in enumerate(layer_idxs):
#         print(layer_idx)
#         if layer_idx.item() not in layer_groups:
#             layer_groups[layer_idx.item()] = ([], [])
#         layer_groups[layer_idx.item()][0].append(neuron_idxs[i])
#         layer_groups[layer_idx.item()][1].append(assigned_values[i])

#     # Register one hook per unique layer, assigning multiple neurons in parallel
#     for layer_idx, (neuron_idx_list, assigned_value_list) in layer_groups.items():
#         neuron_idx_tensor = torch.tensor(neuron_idx_list, device="cuda")
#         assigned_value_tensor = torch.tensor(assigned_value_list, device="cuda")

#         # Register a single hook for each layer that applies all neuron changes in one batch
#         hook = model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(
#             patch(layer_idx, neuron_idx_tensor, assigned_value_tensor)
#         )
#         hooks.append(hook)

#     print(f"Successfully registered hooks for {len(layer_groups)} layers with GPU batch processing.")
#     return model, hooks  




def assign_activations_to_neurons_new(model, config):
    """
    Modify the activation coefficients for specific neurons in different layers.
    Each entry in the config contains a tuple (layer_idx, neuron_idx, assigned_value).
    """
    config_path = config['neuron_configs_path']  # csv file path
    hook_timesteps = config["hook_timesteps"]

    df_neurons = pd.read_csv(config_path)
    neuron_configs = df_neurons.to_records(index=False).tolist()  # List of (layer_idx, neuron_idx, assigned_value)

    # Batch processing 
    layer_idxs = torch.tensor([cfg[0] for cfg in neuron_configs], device="cuda")
    neuron_idxs = torch.tensor([cfg[1] for cfg in neuron_configs], device="cuda")
    assigned_values = torch.tensor([cfg[2] for cfg in neuron_configs], device="cuda")

    def patch(layer_idx, neuron_idx_tensor, assigned_value_tensor):
        def hook(module, input):
            """
            Forward hook to assign specific values to multiple neurons' activation coefficients 
            at the **input** to mlp.down_proj.
            """
            with torch.no_grad():
                assigned_value_tensor_casted = assigned_value_tensor.to(input[0].dtype)

                # Assign the modified activation values
                input[0][:, hook_timesteps, neuron_idx_tensor] = assigned_value_tensor_casted
                # print(f"Assigned values to neurons in layer {layer_idx} at input to mlp.down_proj")

            return input  

        return hook

    hooks = []

    # Group by layer for parallel assignment within each layer
    layer_groups = {}
    for i, layer_idx in enumerate(layer_idxs):
        if layer_idx.item() not in layer_groups:
            layer_groups[layer_idx.item()] = ([], [])
        layer_groups[layer_idx.item()][0].append(neuron_idxs[i])
        layer_groups[layer_idx.item()][1].append(assigned_values[i])

    # Register one hook per unique layer, assigning multiple neurons in parallel
    for layer_idx, (neuron_idx_list, assigned_value_list) in layer_groups.items():
        neuron_idx_tensor = torch.tensor(neuron_idx_list, device="cuda")
        assigned_value_tensor = torch.tensor(assigned_value_list, device="cuda")

        # Register a single hook for each layer that applies all neuron changes in one batch
        hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            patch(layer_idx, neuron_idx_tensor, assigned_value_tensor)
        )
        hooks.append(hook)

    # print(f"Successfully registered hooks for {len(layer_groups)} layers at input to mlp.down_proj.")
    return model, hooks  




def assign_activations_to_neurons_full(model, config):
    """
    Modify activation coefficients for specific neurons in different layers.
    Supports both LLama and GPT2 architectures.
    """
    config_path = config['neuron_configs_path']
    hook_timesteps = config["hook_timesteps"]

    df_neurons = pd.read_csv(config_path)
    neuron_configs = df_neurons.to_records(index=False).tolist()  # List of (layer_idx, neuron_idx, assigned_value)

    # Batch processing
    layer_idxs = torch.tensor([cfg[0] for cfg in neuron_configs], device="cuda")
    neuron_idxs = torch.tensor([cfg[1] for cfg in neuron_configs], device="cuda")
    assigned_values = torch.tensor([cfg[2] for cfg in neuron_configs], device="cuda")

    def patch(layer_idx, neuron_idx_tensor, assigned_value_tensor):
        def hook(module, input):
            with torch.no_grad():
                assigned_value_tensor_casted = assigned_value_tensor.to(input[0].dtype)
                input[0][:, hook_timesteps, neuron_idx_tensor] = assigned_value_tensor_casted
                # print(f"Assigned values to neurons in layer {layer_idx} at input to mlp.down_proj")
            return input
        return hook

    hooks = []
    layer_groups = defaultdict(lambda: ([], []))
    for i, layer_idx in enumerate(layer_idxs):
        layer_groups[layer_idx.item()][0].append(neuron_idxs[i])
        layer_groups[layer_idx.item()][1].append(assigned_values[i])

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Llama-style (model.model.layers)
        layer_accessor = lambda idx: model.model.layers[idx].mlp.down_proj
        print("Detected Llama-style model.")
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT2-style (model.transformer.h)
        layer_accessor = lambda idx: model.transformer.h[idx].mlp.c_proj
        print("Detected GPT2-style model.")
    else:
        raise ValueError("Unsupported model structure. Cannot find layers.")

    # Register one hook per unique layer
    for layer_idx, (neuron_idx_list, assigned_value_list) in layer_groups.items():
        neuron_idx_tensor = torch.tensor(neuron_idx_list, device="cuda")
        assigned_value_tensor = torch.tensor(assigned_value_list, device="cuda")

        try:
            layer = layer_accessor(layer_idx)
            hook = layer.register_forward_pre_hook(
                patch(layer_idx, neuron_idx_tensor, assigned_value_tensor)
            )
            hooks.append(hook)
        except Exception as e:
            print(f"Failed to register hook for layer {layer_idx}: {e}")

    # print(f"Successfully registered hooks for {len(hooks)} layers.")
    return model, hooks




# def assign_activations_to_neurons(model, config):
#     """
#     Modify the activation coefficients for multiple neurons in different layers. Each entry in the config
#     contains a tuple (layer_idx, neuron_idx, assigned_value), and all modifications happen in one hook.
    
#     Args:
#         model: The transformer model to modify.
#         config: Dictionary containing 'neuron_configs' as a list of tuples (layer_idx, neuron_idx, assigned_value).
    
#     Returns:
#         model: The model with the hook registered.
#         hooks: A list of hook handles for cleanup later.
#     """
#     neuron_configs = config['neuron_configs']  # List of (layer_idx, neuron_idx, assigned_value)
#     hooks = []  # List to store hook handles for cleanup
    
#     # Define hook function to modify activations
#     def modify_activation(module, input, output):
#         """
#         Forward hook to assign specific values to the activation coefficients for the specified neurons.
#         """
#         print(f"Modifying activation values for multiple neurons...")
#         print(f"Output shape: {output.shape}")

#         with torch.no_grad():
#             output_mod = output.clone()  # Clone the output tensor to modify it

#             # Iterate over the neuron configurations
#             for (layer_idx, neuron_idx, assigned_value) in neuron_configs:
#                 if neuron_idx < output.shape[-1]:
#                     print(f"Assigning value {assigned_value} to neuron {neuron_idx} at layer {layer_idx}...")
#                     # Assign the specific value to the neuron activation
#                     output_mod[:, -1, neuron_idx] = assigned_value
#                 else:
#                     print(f"Neuron index {neuron_idx} is out of bounds for the output tensor with dimension {output.shape[-1]}")

#         return output_mod  # Return the modified output tensor

#     # Register a hook for each layer in neuron_configs
#     for (layer_idx, neuron_idx, _) in neuron_configs:
#         print(f"Registering hook for layer {layer_idx} and neuron {neuron_idx}...")
#         hook = model.blocks[layer_idx].mlp.hook_post.register_forward_hook(modify_activation)
#         hooks.append(hook)

#     print("Hooks registered successfully.")
#     return model, hooks  # Return the model and list of hooks for cleanup later



           