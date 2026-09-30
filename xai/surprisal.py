import csv
import io
import math
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SURPRISAL_MODELS: dict[str, str] = {
    "openai-gpt": "openai-community/openai-gpt",
    "gpt2": "openai-community/gpt2",
    "gpt2-large": "openai-community/gpt2-large",
    "gemma-3-270m": "google/gemma-3-270m",
    "gemma-3-1b-pt": "google/gemma-3-1b-pt",
    "gemma-3-4b-pt": "google/gemma-3-4b-pt",
    "bloom-560m": "bigscience/bloom-560m",
    "bloom-1b7": "bigscience/bloom-1b7",
    "bloom-3b": "bigscience/bloom-3b",
    "xglm-564M": "facebook/xglm-564M",
    "xglm-1.7B": "facebook/xglm-1.7B",
    "xglm-2.9B": "facebook/xglm-2.9B",
    "german-gpt2": "dbmdz/german-gpt2",
    "german-gpt2-larger": "stefan-it/german-gpt2-larger",
    "gerpt2": "benjamin/gerpt2",
    "gerpt2-large": "benjamin/gerpt2-large",
    "gpt2-chinese-cluecorpussmall": "uer/gpt2-chinese-cluecorpussmall",
    "radlab/polish-gpt2-small-v2": "radlab/polish-gpt2-small-v2",
}

DEFAULT_MODEL_KEY = "gpt2"
SPACE_REPLACEMENT = "\u2423"
BLOCK_CHAR = "\u2588"


def _enable_model_attention_outputs(model: object) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    if hasattr(config, "output_attentions"):
        config.output_attentions = True
    if hasattr(config, "return_dict"):
        config.return_dict = True


def resolve_model(model_key: str) -> str:
    if model_key not in SURPRISAL_MODELS:
        models_list = "\n ".join(SURPRISAL_MODELS.keys())
        raise ValueError(
            f"ERROR: Model {model_key} is not supported. Choose one from: \n {models_list}"
        )
    return SURPRISAL_MODELS[model_key]


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_key: str,
    device: torch.device | None = None,
    model_cache: dict[tuple[str, str], object] | None = None,
    tokenizer_cache: dict[str, object] | None = None,
) -> tuple[object, object, str, torch.device]:
    model_name = resolve_model(model_key)
    device = resolve_device(str(device) if device is not None else None)

    cache_key = ("surprisal", model_name)
    if model_cache is not None and cache_key in model_cache:
        model = model_cache[cache_key]
        if hasattr(model, "to"):
            model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        _enable_model_attention_outputs(model)
        model.to(device)
        if model_cache is not None:
            model_cache[cache_key] = model
    _enable_model_attention_outputs(model)

    if tokenizer_cache is not None and model_name in tokenizer_cache:
        tokenizer = tokenizer_cache[model_name]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer_cache is not None:
            tokenizer_cache[model_name] = tokenizer

    return model, tokenizer, model_name, device


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_text(
    input_text: str,
    nt: int,
    model: object,
    tokenizer: object,
    device: torch.device,
    temperature: float = 1.0,
    topk: int = 50,
) -> str:
    input_tokens = tokenizer(input_text, return_tensors="pt").to(device)
    output_tokens = model.generate(
        **input_tokens,
        max_length=nt,
        temperature=temperature,
        top_k=topk,
        repetition_penalty=1.0,
        do_sample=True,
    )
    output_text = tokenizer.batch_decode(output_tokens)[0]
    return output_text


def compute_surprisal(
    input_text: str,
    model: object,
    tokenizer: object,
    device: torch.device,
) -> tuple[list[str], list[float]]:
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_tokens, labels=input_tokens)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)
    shifted_tokens = input_tokens[..., 1:]
    target_log_probs = log_probs[:, :-1].gather(2, shifted_tokens.unsqueeze(-1)).squeeze(-1)

    log2 = torch.log(torch.tensor(2.0, device=log_probs.device))
    surprisals = -target_log_probs / log2

    first_token = input_tokens[0, 0].item()
    first_token_surprisal = (-log_probs[0, 0, first_token] / log2).item()

    token_ids = input_tokens.squeeze().tolist()
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    decoded_tokens = [tokenizer.decode([tok]) for tok in token_ids]

    surprisal_values = [first_token_surprisal] + surprisals.squeeze(0).tolist()
    return decoded_tokens, surprisal_values


def topn_tokens(
    input_text: str,
    n: int,
    model: object,
    tokenizer: object,
    device: torch.device,
) -> list[tuple[str, float]]:
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits
    last_token_logits = predictions[:, -1, :]
    probabilities = F.softmax(last_token_logits, dim=-1)
    topn_probs, topn_token_ids = torch.topk(probabilities, n)
    surprisals = (-torch.log2(topn_probs)).tolist()
    tokens = [tokenizer.decode([token_id]) for token_id in topn_token_ids[0]]
    return list(zip(tokens, surprisals[0]))


def pretty_print_surprisals(items: Sequence[dict[str, object]], output: io.TextIOBase) -> None:
    item_max = len("item")
    idx_max = len("idx")
    token_max = len("token")
    surprisal_max = len("surprisal (bits)")
    for item in items:
        for idx, (token, surprisal) in enumerate(item["surprisals"]):
            item_max = max(item_max, len(str(item["item"])))
            idx_max = max(idx_max, len(str(idx + 1)))
            token_max = max(token_max, len(token))
            if not math.isnan(surprisal):
                surprisal_max = max(surprisal_max, surprisal)

    output.write(
        "%s %s %s: %s\n"
        % (
            "Item".rjust(item_max),
            "Idx".rjust(idx_max),
            "Token".rjust(token_max),
            "Surprisal (bits)",
        )
    )

    for item in items:
        for idx, (token, surprisal) in enumerate(item["surprisals"]):
            if math.isnan(surprisal):
                sp = ""
            else:
                sp = round(surprisal) * BLOCK_CHAR
            output.write(
                "%s %s %s: %s %s\n"
                % (
                    str(item["item"]).rjust(item_max),
                    str(idx + 1).rjust(idx_max),
                    token.replace(" ", SPACE_REPLACEMENT).rjust(token_max),
                    sp.ljust(round(surprisal_max)),
                    ("%.1f" % (surprisal,)).rjust(5),
                )
            )
    output.flush()


def write_surprisal_csv(items: Sequence[dict[str, object]], output: io.TextIOBase) -> None:
    csvwriter = csv.writer(output, lineterminator="\n")
    csvwriter.writerow(["item", "idx", "token", "surprisal"])
    for item in items:
        for idx, (token, surprisal) in enumerate(item["surprisals"]):
            csvwriter.writerow([item["item"], idx + 1, token.strip(), surprisal])


def pretty_print_topn(items: Sequence[dict[str, object]], output: io.TextIOBase) -> None:
    item_max = len("item")
    text_max = len("text")
    token_max = len("token")
    rank_max = len("rank")
    surprisal_max = len("surprisal (bits)")
    for item in items:
        for rank, (token, surprisal) in enumerate(item["topn"]):
            item_max = max(item_max, len(str(item["item"])))
            text_max = max(text_max, len(str(item["text"])))
            token_max = max(token_max, len(token))
            rank_max = max(rank_max, len(str(rank)))
            if not math.isnan(surprisal):
                surprisal_max = max(surprisal_max, surprisal)

    output.write(
        "%s %s %s %s: %s\n"
        % (
            "Item".rjust(item_max),
            "Text".rjust(text_max),
            "Token".rjust(token_max),
            "Rank".rjust(rank_max),
            "Surprisal (bits)",
        )
    )
    for item in items:
        for rank, (token, surprisal) in enumerate(item["topn"]):
            if math.isnan(surprisal):
                sp = ""
            else:
                sp = round(surprisal) * BLOCK_CHAR
            output.write(
                "%s %s %s %s: %s %s\n"
                % (
                    str(item["item"]).rjust(item_max),
                    item["text"].rjust(text_max),
                    token.replace(" ", SPACE_REPLACEMENT).rjust(token_max),
                    str(rank + 1).rjust(rank_max),
                    sp.ljust(round(surprisal_max)),
                    ("%.1f" % (surprisal,)).rjust(5),
                )
            )
    output.flush()


def write_topn_csv(items: Sequence[dict[str, object]], output: io.TextIOBase) -> None:
    csvwriter = csv.writer(output, lineterminator="\n")
    csvwriter.writerow(["item", "text", "token", "rank", "surprisal"])
    for item in items:
        for rank, (token, surprisal) in enumerate(item["topn"]):
            csvwriter.writerow([item["item"], item["text"], token.strip(), rank + 1, surprisal])
