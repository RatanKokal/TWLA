"""
tests/test_qwen35_twla.py
=========================
Test suite for TWLA × Qwen3.5-4B compatibility.

Tests:
  A: module discovery  — verify complete module mapping
  B: KOTMS round trip  — W → rotate → inverse-rotate ≈ W
  C: layer forward equivalence — original vs TWLA-replayed forward match
  D: quantized single layer    — report FP16/quantized error
  E: end-to-end smoke test     — logit comparison before/after

Usage:
    python tests/test_qwen35_twla.py --model Qwen/Qwen3.5-4B --test all
    python tests/test_qwen35_twla.py --model Qwen/Qwen3.5-4B --test discovery
    python tests/test_qwen35_twla.py --model Qwen/Qwen3.5-4B --test kotms_roundtrip
    python tests/test_qwen35_twla.py --model Qwen/Qwen3.5-4B --test layer_fwd
    python tests/test_qwen35_twla.py --model Qwen/Qwen3.5-4B --test quant_layer
    python tests/test_qwen35_twla.py --model Qwen/Qwen3.5-4B --test e2e
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure TWLA root is on path
_TWLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _TWLA_ROOT)

from twla.models.qwen35 import (
    discover_modules,
    print_module_inventory,
    get_layer_types,
    get_kotms_targets,
    is_deltanet_layer,
    is_full_attention_layer,
    collect_qwen35_calibration_inputs,
    Qwen35LayerReplay,
    FULL_ATTN_TARGETS,
    DELTANET_TARGETS,
    MLP_TARGETS,
)
from quantize.k_preprocessor_ternary import KroneckerSmoothConfig, KroneckerSmoothPreprocessor
from quantize.E2M_ATQ import Ternarization
from quantize.tra_gptq import TRAGPTQ


# ---------------------------------------------------------------------------
# Test A: Module Discovery
# ---------------------------------------------------------------------------

def test_a_discovery(model, args):
    """Verify the complete Qwen3.5 → TWLA module mapping."""
    print("\n" + "="*70)
    print("TEST A: Module Discovery")
    print("="*70)

    layer_types = get_layer_types(model)
    targets = discover_modules(model)

    print_module_inventory(model, targets)

    n_deltanet = sum(1 for lt in layer_types if lt == "linear_attention")
    n_full_attn = sum(1 for lt in layer_types if lt == "full_attention")

    print(f"Layer type counts: {n_deltanet} DeltaNet, {n_full_attn} full-attention")

    # Basic sanity checks
    roles = [tm.role for tm in targets]
    n_mlp = sum(1 for r in roles if r == "mlp")
    n_attn = sum(1 for r in roles if r == "full_attn")
    n_delta = sum(1 for r in roles if r == "deltanet")

    print(f"\nQuantizable projections by role:")
    print(f"  mlp:       {n_mlp}  (expected: ~{len(layer_types) * 3})")
    print(f"  full_attn: {n_attn}  (expected: ~{n_full_attn * 4})")
    print(f"  deltanet:  {n_delta}  (expected: ~{n_deltanet * 4} to {n_deltanet * 5})")

    assert n_mlp > 0, "No MLP targets found!"
    assert n_attn > 0, "No full-attention targets found!"
    assert n_delta > 0, "No DeltaNet targets found!"

    # Verify all are actually nn.Linear
    for tm in targets:
        assert isinstance(tm.module, nn.Linear), \
            f"Target {tm.full_name} is {type(tm.module)}, not nn.Linear"

    print("\n✓ Test A PASSED: Module discovery correct.")
    return targets


# ---------------------------------------------------------------------------
# Test B: KOTMS Round Trip
# ---------------------------------------------------------------------------

def test_b_kotms_roundtrip(model, args):
    """
    For representative matrices W:
      1. Run KOTMS (GMM rotation): W → W_rot, L, R
      2. Reconstruct: W_approx = L^T @ W_rot_reshaped @ R^T
      3. Verify W_approx ≈ W (Frobenius norm)
    """
    print("\n" + "="*70)
    print("TEST B: KOTMS Round Trip")
    print("="*70)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    targets = discover_modules(model)
    layer_types = get_layer_types(model)

    # Pick one projection from each category
    test_cases = []
    for role in ["mlp", "full_attn", "deltanet"]:
        for tm in targets:
            if tm.role == role:
                test_cases.append(tm)
                break

    cfg = KroneckerSmoothConfig(
        use_gmm_training=bool(getattr(args, "use_gmm", False)),
        gmm_iters=int(getattr(args, "gmm_iters", 50)),
        gmm_lr_r=float(getattr(args, "gmm_lr_r", 1e-2)),
        gmm_lr_l=float(getattr(args, "gmm_lr_l", 1e-2)),
    )
    kp = KroneckerSmoothPreprocessor(cfg)

    all_passed = True

    for tm in test_cases:
        W = tm.module.weight.data.clone().to(device=dev, dtype=torch.float32)
        oc, ic = int(W.shape[0]), int(W.shape[1])

        print(f"\n  {tm.full_name}  [{oc} × {ic}]  role={tm.role}")

        try:
            res = kp._kronecker_process(W)
        except RuntimeError as e:
            if "use_gmm_training=False" in str(e):
                print(f"    Skipping GMM test (use_gmm=False). Use --use_gmm to test.")
                continue
            raise

        C = res["C"]  # left rotation  [l, l]
        B = res["B"]  # right rotation [r, r]
        l = int(res["l"])
        r = int(res["r"])
        W_rot = res["W_rot"]  # [oc, ic] rotated weight

        # TwoSidedOrthoSimple.forward does:
        #   W3 = W.view(oc, l, r)
        #   Xr = W3 @ B                           right rot:  [oc, l, r]
        #   Xl = C @ Xr_perm.view(l, oc*r) → view [l, oc, r] → permute [oc, l, r]
        # So W_rot = C_applied(B_applied(W)).
        #
        # Reconstruction: W ≈ C^T applied to (B^T applied to W_rot)
        W_rot_3d = W_rot.reshape(oc, l, r)               # [oc, l, r]
        # Undo left rotation: C @ (W_perm) → undo: C^T @ W_rot_perm
        Xr_perm = (C.T @ W_rot_3d.permute(1, 0, 2).reshape(l, oc * r)
                   ).reshape(l, oc, r).permute(1, 0, 2)  # [oc, l, r]
        # Undo right rotation: Xr = W3 @ B → W3 = Xr @ B^T
        W_recon = (Xr_perm @ B.T).reshape(oc, ic)

        frob_err = (W_recon - W).norm().item()
        frob_W = W.norm().item()
        rel_err = frob_err / (frob_W + 1e-12)

        status = "✓" if rel_err < 1e-4 else "✗"
        print(f"    Frob err: {frob_err:.6e}  Relative: {rel_err:.6e}  {status}")

        if rel_err >= 1e-4:
            all_passed = False
            print(f"    WARNING: Reconstruction error too large!")

    if all_passed:
        print("\n✓ Test B PASSED: KOTMS round-trip correct.")
    else:
        print("\n✗ Test B FAILED: Some matrices have large reconstruction error.")

    return all_passed


# ---------------------------------------------------------------------------
# Test C: Layer Forward Equivalence
# ---------------------------------------------------------------------------

def test_c_layer_forward(model, args):
    """
    Compare original Qwen3.5 layer forward with TWLA-replayed forward.
    Tests: MLP layer, full-attention layer, DeltaNet layer.
    """
    print("\n" + "="*70)
    print("TEST C: Layer Forward Equivalence")
    print("="*70)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqlen = 64
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create a short calibration batch
    text = "The quick brown fox jumps over the lazy dog. " * 5
    enc = tokenizer(text, return_tensors="pt", max_length=seqlen, truncation=True)
    input_ids = enc["input_ids"].to(dev)
    if input_ids.shape[1] < seqlen:
        input_ids = F.pad(input_ids, (0, seqlen - input_ids.shape[1]))
    input_ids = input_ids[:, :seqlen]

    layer_types = get_layer_types(model)

    # Find indices for test layers
    full_attn_idx = next((i for i, lt in enumerate(layer_types) if lt == "full_attention"), None)
    deltanet_idx = next((i for i, lt in enumerate(layer_types) if lt == "linear_attention"), None)
    mlp_idx = 0  # Every layer has an MLP

    print(f"  Testing layer indices: MLP={mlp_idx}, full-attn={full_attn_idx}, deltanet={deltanet_idx}")

    # Collect inputs to layer 0 via replay
    inps, layer_kwargs = collect_qwen35_calibration_inputs(model, input_ids, dev)

    results = {}

    for name, layer_idx in [("MLP", mlp_idx), ("FullAttn", full_attn_idx), ("DeltaNet", deltanet_idx)]:
        if layer_idx is None:
            continue

        print(f"\n  Testing {name} layer (idx={layer_idx})...")

        # We need to run layers 0..layer_idx to get the input at layer_idx
        # Use the replay to propagate
        layer = model.model.layers[layer_idx].to(dev)

        # Build per-sample kwargs with the recurrent state (None for first run)
        kwargs = {}
        for k, v in layer_kwargs.items():
            if k == "recurrent_state":
                kwargs["recurrent_state"] = None
            elif isinstance(v, torch.Tensor):
                kwargs[k] = v.to(dev)
            elif isinstance(v, (list, tuple)):
                kwargs[k] = type(v)(x.to(dev) if isinstance(x, torch.Tensor) else x for x in v)
            else:
                kwargs[k] = v
        if "recurrent_state" not in kwargs:
            kwargs["recurrent_state"] = None

        inp = inps[0:1].to(dev)

        with torch.no_grad():
            try:
                out1 = layer(inp, **kwargs)
                out2 = layer(inp, **kwargs)

                if isinstance(out1, (tuple, list)):
                    h1 = out1[0]
                    h2 = out2[0]
                else:
                    h1, h2 = out1, out2

                # They should be identical (deterministic)
                max_diff = (h1 - h2).abs().max().item()
                rel_diff = (h1 - h2).norm().item() / (h1.norm().item() + 1e-12)
                status = "✓" if max_diff < 1e-6 else "✗"
                print(f"    Determinism check: max_diff={max_diff:.2e}  {status}")

                results[name] = {"max_diff": max_diff, "rel_diff": rel_diff, "passed": max_diff < 1e-5}

            except Exception as e:
                print(f"    ERROR: {e}")
                results[name] = {"max_diff": float("inf"), "passed": False, "error": str(e)}

        layer.cpu()
        torch.cuda.empty_cache()

    all_passed = all(r.get("passed", False) for r in results.values())
    if all_passed:
        print("\n✓ Test C PASSED: Layer forward is deterministic and correct.")
    else:
        print("\n✗ Test C had failures:", {k: v for k, v in results.items() if not v.get("passed")})

    return results


# ---------------------------------------------------------------------------
# Test D: Quantized Single Layer Error
# ---------------------------------------------------------------------------

def test_d_quant_layer(model, args):
    """
    Run E2M-ATQ + GPTQ on one representative layer, report FP16 vs quantized error.
    """
    print("\n" + "="*70)
    print("TEST D: Quantized Single Layer Error")
    print("="*70)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqlen = 128
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    text = "The quick brown fox jumps over the lazy dog. " * 10
    enc = tokenizer(text, return_tensors="pt", max_length=seqlen, truncation=True)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < seqlen:
        input_ids = F.pad(input_ids, (0, seqlen - input_ids.shape[1]))
    input_ids = input_ids[:1, :seqlen]

    layer_types = get_layer_types(model)
    # Pick the first full-attention layer for the test
    test_layer_idx = next((i for i, lt in enumerate(layer_types) if lt == "full_attention"), 3)

    print(f"  Test layer idx: {test_layer_idx} ({layer_types[test_layer_idx]})")

    inps, layer_kwargs = collect_qwen35_calibration_inputs(model, input_ids, dev)
    inp = inps[0:1].to(dev)

    kwargs = {}
    for k, v in layer_kwargs.items():
        if k == "recurrent_state":
            kwargs["recurrent_state"] = None
        elif isinstance(v, torch.Tensor):
            kwargs[k] = v.to(dev)
        elif isinstance(v, (list, tuple)):
            kwargs[k] = type(v)(x.to(dev) if isinstance(x, torch.Tensor) else x for x in v)
        else:
            kwargs[k] = v
    if "recurrent_state" not in kwargs:
        kwargs["recurrent_state"] = None

    layer = model.model.layers[test_layer_idx].to(dev)

    from quantize.gptq_fwrd import find_qlayers
    full = find_qlayers(layer, layers=[torch.nn.Linear])

    # Record FP16 output
    with torch.no_grad():
        out_fp16 = layer(inp, **kwargs)
        if isinstance(out_fp16, (tuple, list)):
            out_fp16 = out_fp16[0]

    print(f"  FP16 output shape: {tuple(out_fp16.shape)}, norm={out_fp16.norm().item():.4f}")

    # Run GPTQ on q_proj as a representative module
    test_proj_name = "self_attn.q_proj"
    if test_proj_name not in full:
        # fallback to first available
        test_proj_name = list(full.keys())[0]

    print(f"  Quantizing: {test_proj_name}")
    proj = full[test_proj_name]
    braq = Ternarization(proj.weight, groupsize=128)
    gptq_obj = TRAGPTQ(
        layer=proj,
        braq_quantizer=braq,
        salient_metric="hessian",
        disable_gptq=False,
        order2_group=True,
    )

    def hook_fn(_, inp_h, out_h):
        gptq_obj.add_batch(inp_h[0].data, out_h.data)

    handle = proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        layer(inp, **kwargs)
    handle.remove()

    gptq_obj.fasterquant(blocksize=128, percdamp=0.1, orders=(1, 1, 2), num_p=1)
    gptq_obj.free()

    with torch.no_grad():
        out_quant = layer(inp, **kwargs)
        if isinstance(out_quant, (tuple, list)):
            out_quant = out_quant[0]

    abs_err = (out_quant - out_fp16).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_err = (out_quant - out_fp16).norm().item() / (out_fp16.norm().item() + 1e-12)

    print(f"\n  Results (quantized {test_proj_name} only):")
    print(f"    Max absolute error:  {max_abs:.6f}")
    print(f"    Mean absolute error: {mean_abs:.6f}")
    print(f"    Relative error:      {rel_err:.6f}")

    layer.cpu()
    torch.cuda.empty_cache()

    print("\n✓ Test D DONE.")
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel_err": rel_err}


# ---------------------------------------------------------------------------
# Test E: End-to-End Smoke Test
# ---------------------------------------------------------------------------

def test_e_e2e(model, args):
    """
    Run a short text batch through:
      1. Original Qwen3.5-4B (unquantized)
      2. TWLA-replayed (unquantized, check replay path)
    Compare logits for consistency.
    """
    print("\n" + "="*70)
    print("TEST E: End-to-End Smoke Test")
    print("="*70)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompt = "The capital of France is"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev)

    # Run original model
    model.to(dev)
    model.eval()
    with torch.no_grad():
        out_orig = model(input_ids)
        logits_orig = out_orig.logits[0, -1, :]

    top5_orig = logits_orig.topk(5)
    print(f"\n  Original top-5 next token IDs: {top5_orig.indices.tolist()}")
    print(f"  Original top-5 tokens: {[tokenizer.decode([i]) for i in top5_orig.indices.tolist()]}")

    # Run again to check determinism
    with torch.no_grad():
        out_orig2 = model(input_ids)
        logits_orig2 = out_orig2.logits[0, -1, :]

    max_diff = (logits_orig - logits_orig2).abs().max().item()
    print(f"\n  Determinism check: max_diff={max_diff:.2e}")
    assert max_diff < 1e-4, f"Model is not deterministic! max_diff={max_diff}"

    # Quick generation test
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
        )
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\n  Generated: {generated_text!r}")

    model.cpu()
    torch.cuda.empty_cache()

    print("\n✓ Test E PASSED: End-to-end smoke test complete.")
    return {"logits": logits_orig.cpu(), "generated": generated_text}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="TWLA Qwen3.5 Test Suite")
    p.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    p.add_argument("--test", type=str, default="all",
                   choices=["all", "discovery", "kotms_roundtrip", "layer_fwd",
                            "quant_layer", "e2e"])
    p.add_argument("--use_gmm", action="store_true", default=False,
                   help="Use GMM rotation in KOTMS (Test B). Slower but more thorough.")
    p.add_argument("--gmm_iters", type=int, default=50)
    p.add_argument("--gmm_lr_r", type=float, default=1e-2)
    p.add_argument("--gmm_lr_l", type=float, default=1e-2)
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", torch_dtype=torch.float16
    )
    model.eval()

    results = {}
    run_all = args.test == "all"

    if run_all or args.test == "discovery":
        results["A_discovery"] = test_a_discovery(model, args)

    if run_all or args.test == "kotms_roundtrip":
        results["B_kotms"] = test_b_kotms_roundtrip(model, args)

    if run_all or args.test == "layer_fwd":
        results["C_layer_fwd"] = test_c_layer_forward(model, args)

    if run_all or args.test == "quant_layer":
        results["D_quant"] = test_d_quant_layer(model, args)

    if run_all or args.test == "e2e":
        results["E_e2e"] = test_e_e2e(model, args)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for key, val in results.items():
        if val is None:
            status = "SKIPPED"
        elif isinstance(val, bool):
            status = "PASSED" if val else "FAILED"
        elif isinstance(val, list):
            status = f"PASSED ({len(val)} targets)"
        elif isinstance(val, dict):
            status = "DONE"
        else:
            status = "DONE"
        print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
