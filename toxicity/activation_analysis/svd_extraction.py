import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

def get_svd_u_vec(model, toxic_vector, topk_sorted_score, U_idx):
    """
    Get the SVD U vector.
    """
    # Ensure toxic_vector is on the same device as model weights
    device = model.transformer.h[0].mlp.c_proj.weight.device
    toxic_vector = toxic_vector.to(device)

    scores = []
    for layer in range(model.config.n_layer):
        mlp_outs = model.transformer.h[layer].mlp.c_proj.weight  # [d_mlp, d_model]
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
        model.transformer.h[x[2]].mlp.c_proj.weight[x[1]]
        for x in sorted_scores[:topk_sorted_score]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = torch.stack(top_vecs)
    print(f"Stacked top vectors shape: {_top_vecs.shape}")

    svd = torch.linalg.svd(_top_vecs.transpose(0, 1))
    svd_U = svd.U.transpose(0, 1)
    print(f"SVD vector shape: {svd_U[0].shape}")
    return svd_U[U_idx]


# ==== MAIN SCRIPT ====
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").cuda()
model.eval()

toxic_vector = torch.load("/data/kebl6672/dpo-toxic-general/ignore/gpt2_probe.pt")

topk_sorted_score = 128

for u_idx in range(3):
    u_vector = get_svd_u_vec(model, toxic_vector, topk_sorted_score, u_idx)
    save_path = f"/data/kebl6672/dpo-toxic-general/ignore/svd_{u_idx}.pt"
    torch.save(u_vector.cpu(), save_path)  
    print(f"Saved SVD vector {u_idx}.")
