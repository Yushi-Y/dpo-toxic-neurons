"""
Intervention functionalities.
"""
from tqdm import tqdm
import torch
from utils import verbose_print, VERBOSE

device = "cuda"


def get_prompts(model, data, config):
    """
    Dummy intervention.
    """
    return {
        "pred_tokens": data["prompt_input_ids"],
        "pred_text": data["prompts"],
    }


def get_gold(model, data, config):
    """
    Dummy intervention.
    """
    return {
        "pred_tokens": data["gold_input_ids"],
        "pred_text": data["gold"],
    }



def generate_default(model, data, config):
    """
    Do not intervene. Added attention_mask for models that require it.
    """
    batch_size = config["batch_size"]
    pad_token_id = model.tokenizer.pad_token_id
    all_output = []
    all_output_text = []
    # Bad words
    # bad_words = ["shit", "fuck", "fucked", "fucking", "fucks", "shitty", "crap", "damn", "asshole", 
                #  "ass", "cunt", "bitch", "dick", "bullshit", "whore", "slut", "hell", "bustard", 
                #  "cum", "nuts", "sexual", "sex", "porn", "misogyn", "death", "extermination", 
                #  "corpse", "decap", "offending", "horrible", "junk", "garbage", "trash", "extermination", 
                #  "decap", "idiot", "stupid", "moron", "ridiculous", "absurd", "ludicrous", "heck"]  
    # bad_words_ids = model.tokenizer(bad_words, add_special_tokens=False).input_ids


    for idx in tqdm(range(0, data["prompt_input_ids"].shape[0], batch_size)):
        batch = data["prompt_input_ids"][idx : idx + batch_size]

        # Only set attention_mask for models that require it
        model_name = model.config._name_or_path.lower()  
        requires_attention_mask = any(name in model_name for name in ["llama", "mistral", "gpt"])

        attention_mask = None
        if requires_attention_mask:
            attention_mask = (batch != pad_token_id).long()

        with torch.inference_mode():
            model.to(device)
            output = model.generate(
                batch.to(device),
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                pad_token_id=pad_token_id,
                # bad_words_ids=bad_words_ids,
                attention_mask=attention_mask.to(device) if attention_mask is not None else None, 
            )

            if VERBOSE:
                model_inputs = {"input_ids": batch.to(device)}
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask.to(device)

                _output = model(**model_inputs)
                logits = _output.logits
                topk = logits.topk(k=5).indices
                verbose_print(model.tokenizer.batch_decode(topk[:, -1, :]))

        output_text = model.tokenizer.batch_decode(
            output, skip_special_tokens=True
        )
        all_output.extend(output)
        all_output_text.extend(output_text)

    return {
        "pred_tokens": torch.stack(all_output, dim=0),
        "pred_text": all_output_text,
    }



# def generate_default(model, data, config):
#     """
#     Do not intervene.
#     """
#     batch_size = config["batch_size"]
#     pad_token_id = model.tokenizer.pad_token_id
#     all_output = []
#     all_output_text = []

#     for idx in tqdm(range(0, data["prompt_input_ids"].shape[0], batch_size)):
#         batch = data["prompt_input_ids"][idx : idx + batch_size]
#         with torch.inference_mode():
#             model.to(device)
#             output = model.generate(
#                 batch.to(device),
#                 max_new_tokens=config["max_new_tokens"],
#                 do_sample=False,
#                 pad_token_id=pad_token_id,
#             )

#             if VERBOSE:
#                 _output = model.forward(batch.to(device))
#                 logits = _output.logits
#                 topk = logits.topk(k=5).indices
#                 verbose_print(model.tokenizer.batch_decode(topk[:, -1, :]))

#         output_text = model.tokenizer.batch_decode(
#             output, skip_special_tokens=True
#         )
#         all_output.extend(output)
#         all_output_text.extend(output_text)

#     return {
#         "pred_tokens": torch.stack(all_output, dim=0),
#         "pred_text": all_output_text,
#     }
