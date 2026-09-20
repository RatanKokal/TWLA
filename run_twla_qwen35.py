"""
run_twla_qwen35.py
==================
Qwen3.5-4B W1.58A16 ternarization pipeline.

This is the main entry point for running the complete TWLA pipeline
(KOTMS → E2M-ATQ → GPTQ) on Qwen3.5-4B.

Pipeline:
  1.  Load Qwen3.5-4B
  2.  Import KOTMS-rotated checkpoint (from scripts/KOTMS_qwen35.py)
  3.  Run E2M-ATQ + GPTQ quantization (via twla/models/qwen35.py)
  4.  (Optional) evaluate perplexity and/or QA tasks
  5.  (Optional) save/load quantized model

Usage:
    # Stage 2-3: quantization
    python run_twla_qwen35.py \\
        --model Qwen/Qwen3.5-4B \\
        --import_rotated outputs/qwen35_4b_rotated.pt \\
        --wbits 2 \\
        --nsamples 128 \\
        --dataset wikitext2 \\
        --dp_cache cache/qwen35 \\
        --save_quant_model outputs/qwen35_4b_w158.pt

    # Inference only (load quantized model):
    python run_twla_qwen35.py \\
        --model Qwen/Qwen3.5-4B \\
        --import_rotated outputs/qwen35_4b_rotated.pt \\
        --load_quant_model outputs/qwen35_4b_w158.pt
"""
from __future__ import annotations

import os
import sys
import copy
import argparse
import random
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure TWLA root is on path
_TWLA_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _TWLA_ROOT)

from datautils import get_loaders
from quantize.quantizer import UniformAffineQuantizer
from twla.models.qwen35 import (
    twla_fwrd_qwen35,
    import_rotated_checkpoint_qwen35,
    get_kotms_targets,
    discover_modules,
    print_module_inventory,
    get_layer_types,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# ---------------------------------------------------------------------------
# QLinear (same as in run_twla.py — needed for import compatibility)
# ---------------------------------------------------------------------------

class QLinear(nn.Linear):
    @torch.no_grad()
    def init_act_quant(self, args) -> None:
        if int(getattr(args, "abits", 16)) >= 16:
            self.act_quant = None
            return
        lac = float(getattr(self, "act_lac", float(getattr(args, "lac", 0.9))))
        self.act_quant = UniformAffineQuantizer(
            n_bits=int(args.abits),
            symmetric=False,
            per_channel_axes=[],
            dynamic=True,
            dynamic_method=str(getattr(args, "a_dynamic_method", "per_token")),
            act_group_size=getattr(args, "act_group_size", None),
            lac=lac,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_shape = x.shape
        if hasattr(self, "L") and self.L is not None:
            device = x.device
            L = self.L.to(device) if self.L.device != device else self.L
            R = self.R.to(device) if self.R.device != device else self.R
            x = x.reshape(-1, self.dim_l, self.dim_r)
            x = L @ x @ R
            x = x.reshape(init_shape)
        if getattr(self, "_cali_enabled", False):
            self.cali_update(x)
        if getattr(self, "act_quant", None) is not None:
            x = self.act_quant(x)
        return super().forward(x)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_quantized_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def load_quantized_model(model, path: str, strict: bool = True):
    sd = torch.load(path, map_location="cpu")
    return model.load_state_dict(sd, strict=bool(strict))


# ---------------------------------------------------------------------------
# Per-block activation bit-width setter
# ---------------------------------------------------------------------------

def set_block_abits(model, args, block_idx: int, abits: int) -> None:
    a2 = copy.copy(args)
    a2.abits = int(abits)
    block = model.model.layers[int(block_idx)]
    for m in block.modules():
        if isinstance(m, QLinear):
            m.init_act_quant(a2)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, args, datasets=("wikitext2",)) -> None:
    from eval_ppl_utils import llama_eval
    dev = "cuda:0"
    for ds in datasets:
        _, testloader = get_loaders(ds, seed=int(args.seed), seqlen=int(args.seqlen), model=args.model)
        llama_eval(model, testloader, dev, ds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="TWLA Qwen3.5-4B W1.58A16 Pipeline")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seqlen", type=int, default=2048)

    p.add_argument("--import_rotated", type=str, required=True,
                   help="Path to KOTMS-rotated checkpoint (from KOTMS_qwen35.py)")

    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)

    p.add_argument("--wbits", type=int, default=2)
    p.add_argument("--symmetric", action="store_true", default=True)
    p.add_argument("--group_size", type=int, default=None)
    p.add_argument("--swc", type=float, default=0.9)
    p.add_argument("--mse", type=float, default=None)
    p.add_argument("--act_order", action="store_true", default=False)

    p.add_argument("--abits", type=int, default=16,
                   help="Activation bits (16 = W1.58A16, no act quantization)")
    p.add_argument("--a_dynamic_method", type=str, default="per_token")
    p.add_argument("--act_group_size", type=int, default=None)
    p.add_argument("--lac", type=float, default=0.9)

    p.add_argument("--save_quant_model", type=str, default=None)
    p.add_argument("--load_quant_model", type=str, default=None)

    p.add_argument("--eval_ppl", action="store_true", default=False)

    p.add_argument("--print_inventory", action="store_true", default=True)

    # GPTQ options
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.1)
    p.add_argument("--num_p", type=int, default=1)
    p.add_argument("--disable_gptq_core", action="store_true", default=False)
    p.add_argument("--order2_group", action="store_true", default=True)
    p.add_argument("--salient_metric", type=str, default="hessian")

    args = p.parse_args()
    make_deterministic(int(args.seed))

    print(f"Loading Qwen3.5 model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
    model.seqlen = int(args.seqlen)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if bool(args.print_inventory):
        print_module_inventory(model)

    # Load KOTMS-rotated checkpoint
    print(f"\nLoading KOTMS checkpoint: {args.import_rotated}")
    filled = import_rotated_checkpoint_qwen35(model, args.import_rotated, args)
    if filled <= 0:
        raise RuntimeError("No layers loaded from rotated checkpoint. Check path and model_type.")
    print(f"Loaded {filled} rotated projections.")

    model.half()
    model.eval()

    trainloader, _ = get_loaders(
        args.dataset,
        nsamples=int(args.nsamples),
        seed=int(args.seed),
        model=args.model,
        seqlen=int(args.seqlen),
    )

    if args.load_quant_model is not None:
        info = load_quantized_model(model, args.load_quant_model, strict=False)
        print(f"Loaded quantized model: {args.load_quant_model}")
        print(f"Missing keys: {len(info.missing_keys)}, Unexpected: {len(info.unexpected_keys)}")
    else:
        print("\nRunning E2M-ATQ + GPTQ quantization on Qwen3.5-4B...")
        twla_fwrd_qwen35(model, tokenizer, trainloader, "cuda", args)
        if args.save_quant_model is not None:
            save_quantized_model(model, args.save_quant_model)
            print(f"Saved quantized model: {args.save_quant_model}")

    # Disable activation quantization for W1.58A16
    if int(getattr(args, "abits", 16)) >= 16:
        print("W1.58A16 mode: no activation quantization.")
        for i in range(len(model.model.layers)):
            set_block_abits(model, args, i, 16)

    model.cuda().half()
    model.eval()

    if bool(args.eval_ppl):
        evaluate_ppl(model, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
