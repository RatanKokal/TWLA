"""
twla/models/qwen35.py
=====================
Qwen3.5-4B compatibility adapter for TWLA W1.58A16 ternarization.

Architecture facts (verified from HuggingFace transformers source):
  - Model class : Qwen3_5ForCausalLM  (text-only) or Qwen3_5ForConditionalGeneration
  - Text backbone: model.model  (Qwen3_5TextModel, which is Qwen3NextModel)
  - Decoder layers: model.model.layers[i]  (32 total for 4B)
  - Layer types: config.text_config.layer_types  ->  list of
       "linear_attention" (DeltaNet, 24 layers) or
       "full_attention"   (softmax attention,  8 layers)
  - Pattern: [DeltaNet, DeltaNet, DeltaNet, FullAttn] × 8

Module names (verified from modular_qwen3_next.py):
  Full-attention layer (layer.self_attn.*):
    q_proj, k_proj, v_proj, o_proj

  GatedDeltaNet layer (layer.linear_attn.*):
    in_proj_qkv, in_proj_z, in_proj_b, (in_proj_a — optional), out_proj

  MLP (layer.mlp.*):
    gate_proj, up_proj, down_proj

  NOT quantized:
    embed_tokens, lm_head, *.norm*, *.conv1d, *.q_norm, *.k_norm

Decoder layer forward signature:
  layer(hidden_states, attention_mask, position_ids,
        past_key_values, recurrent_state, position_embeddings, ...)
  returns (hidden_states, recurrent_state_or_None)
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: which sub-module names to quantize, by role
# ---------------------------------------------------------------------------

#: Target projections inside a **full-attention** block
FULL_ATTN_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)

#: Target projections inside a **Gated DeltaNet / linear-attention** block.
#: in_proj_a may be absent depending on config — checked with hasattr at runtime.
DELTANET_TARGETS: Tuple[str, ...] = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",   # optional — verified at discover time
    "linear_attn.out_proj",
)

#: MLP targets present in every decoder layer
MLP_TARGETS: Tuple[str, ...] = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

#: Names that must NEVER be quantized (matched by substring)
SKIP_SUBSTRINGS: Tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    ".norm",
    "q_norm",
    "k_norm",
    "conv1d",
    "subln",
    "norm_gated",
    "rotary_emb",
)


# ---------------------------------------------------------------------------
# Layer classification helpers
# ---------------------------------------------------------------------------

def get_layer_types(model) -> List[str]:
    """Return the list of layer types from the model config."""
    cfg = model.config
    # For Qwen3_5ForCausalLM the text config is at cfg (it IS the text config)
    # For Qwen3_5ForConditionalGeneration text config is at cfg.text_config
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is None:
        raise AttributeError(
            "Cannot find 'layer_types' in model config. "
            "Ensure you are using a Qwen3.5 model from a recent transformers version."
        )
    return list(layer_types)


def is_deltanet_layer(layer_types: List[str], layer_idx: int) -> bool:
    """Return True if layer_idx is a Gated DeltaNet (linear-attention) layer."""
    lt = layer_types[layer_idx]
    return lt == "linear_attention"


def is_full_attention_layer(layer_types: List[str], layer_idx: int) -> bool:
    """Return True if layer_idx is a full (softmax) attention layer."""
    lt = layer_types[layer_idx]
    return lt == "full_attention"


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------

@dataclass
class TargetModule:
    """Describes a single quantizable projection inside a Qwen3.5 decoder layer."""
    full_name: str           # e.g. "layers.3.self_attn.q_proj"
    layer_idx: int           # decoder layer index
    role: str                # "full_attn", "deltanet", or "mlp"
    proj_name: str           # e.g. "self_attn.q_proj"
    module: nn.Linear        # the actual parameter module


def _resolve_submodule(parent: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    """Follow dotted attribute path from parent; return None if missing."""
    m = parent
    for part in dotted_name.split("."):
        m = getattr(m, part, None)
        if m is None:
            return None
    return m


def _should_skip(full_name: str) -> bool:
    for substr in SKIP_SUBSTRINGS:
        if substr in full_name:
            return True
    return False


def discover_modules(model) -> List[TargetModule]:
    """
    Walk model.model.layers and return every nn.Linear that TWLA should quantize.

    Returns:
        Ordered list of TargetModule, in the order they appear during a forward pass.
    """
    layers = model.model.layers
    layer_types = get_layer_types(model)
    n_layers = len(layers)

    targets: List[TargetModule] = []

    for li in range(n_layers):
        layer = layers[li]
        lt = layer_types[li]

        # Determine which attention-side projections to target
        if lt == "full_attention":
            attn_targets = [p for p in FULL_ATTN_TARGETS]
            role_attn = "full_attn"
        elif lt == "linear_attention":
            attn_targets = []
            for p in DELTANET_TARGETS:
                mod = _resolve_submodule(layer, p)
                if isinstance(mod, nn.Linear):
                    attn_targets.append(p)
                # else: skip (in_proj_a may not exist)
            role_attn = "deltanet"
        else:
            logger.warning("Layer %d has unknown type %r — skipping attn targets", li, lt)
            attn_targets = []
            role_attn = "unknown"

        # MLP targets are the same for every layer
        mlp_target_names = list(MLP_TARGETS)

        for proj_name, role in (
            [(p, role_attn) for p in attn_targets] +
            [(p, "mlp") for p in mlp_target_names]
        ):
            full_name = f"layers.{li}.{proj_name}"
            if _should_skip(full_name):
                continue
            mod = _resolve_submodule(layer, proj_name)
            if not isinstance(mod, nn.Linear):
                logger.debug("Skipping %s — not nn.Linear (%s)", full_name, type(mod))
                continue
            targets.append(TargetModule(
                full_name=full_name,
                layer_idx=li,
                role=role,
                proj_name=proj_name,
                module=mod,
            ))

    return targets


def print_module_inventory(model, targets: Optional[List[TargetModule]] = None) -> None:
    """Print a clean inventory of all quantizable modules."""
    if targets is None:
        targets = discover_modules(model)

    layer_types = get_layer_types(model)
    print(f"\n{'='*76}")
    print(f"{'Qwen3.5 TWLA Module Inventory':^76}")
    print(f"{'='*76}")
    print(f"{'Layer':>5}  {'Type':>12}  {'Role':>8}  {'Projection':<32}  {'Shape'}")
    print(f"{'-'*76}")

    for tm in targets:
        lt = layer_types[tm.layer_idx]
        shape_str = str(tuple(tm.module.weight.shape))
        print(
            f"{tm.layer_idx:>5}  {lt:>12}  {tm.role:>8}  "
            f"{tm.proj_name:<32}  {shape_str}"
        )

    print(f"{'-'*76}")
    print(f"Total quantizable projections: {len(targets)}")
    print(f"{'='*76}\n")


# ---------------------------------------------------------------------------
# KOTMS target filter
# ---------------------------------------------------------------------------

def get_kotms_linear_layers(model) -> List[Tuple[str, nn.Linear]]:
    """
    Return the (name, module) pairs that KOTMS should rotate.

    This replaces the generic `model.model.named_modules()` walk in KOTMS.py
    to avoid accidentally rotating norms, conv1d, etc.
    """
    targets = discover_modules(model)
    return [(tm.full_name, tm.module) for tm in targets]


# ---------------------------------------------------------------------------
# Calibration / Layer-replay for Qwen3.5
# ---------------------------------------------------------------------------

class Qwen35LayerCatcher(nn.Module):
    """
    Wraps decoder layer 0 to intercept the very first set of inputs
    that the model's embedding layer passes through.

    Stores:
      self.inps          — hidden states for every sample in the batch
      self.layer_kwargs  — the other kwargs (attention_mask, position_ids,
                           position_embeddings, recurrent_state, ...)
    """

    def __init__(self, layer: nn.Module, inps: torch.Tensor, cache: Dict):
        super().__init__()
        self._layer = layer
        self._inps = inps
        self._cache = cache

    def forward(self, inp, **kwargs):
        self._inps[self._cache["i"]].copy_(inp.detach())
        # Only save kwargs once (they are identical across all samples)
        if self._cache["kwargs"] is None:
            # Move tensor kwargs to CPU to avoid keeping them pinned on GPU
            saved = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    saved[k] = v.detach().cpu()
                elif isinstance(v, (list, tuple)):
                    saved[k] = type(v)(
                        x.detach().cpu() if isinstance(x, torch.Tensor) else x
                        for x in v
                    )
                else:
                    saved[k] = v
            self._cache["kwargs"] = saved
        self._cache["i"] += 1
        raise StopIteration  # sentinel to break model forward early

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._layer, name)


def collect_qwen35_calibration_inputs(
    model,
    input_ids: torch.Tensor,
    dev: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run just the embedding layer and catch the inputs to decoder layer 0.

    Returns:
        inps        — FloatTensor [nsamples, seqlen, hidden_size]
        layer_kwargs — dict of (attention_mask, position_ids,
                        position_embeddings, recurrent_state, ...)
                       with all tensors on `dev`.
    """
    bs, seqlen = int(input_ids.size(0)), int(input_ids.size(1))

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    hidden = int(model.config.hidden_size)

    inps = torch.zeros((bs, seqlen, hidden), dtype=dtype, device=dev)
    cache: Dict[str, Any] = {"i": 0, "kwargs": None}

    # Move just what we need onto GPU
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # rotary embedding is typically stored on model.model directly
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    if hasattr(model.model, "norm"):
        pass  # keep norm on CPU; we only need it after all layers
    layers[0] = layers[0].to(dev)

    catcher = Qwen35LayerCatcher(layers[0], inps, cache)
    layers[0] = catcher

    for i in range(bs):
        try:
            model(input_ids[i:i + 1].to(dev))
        except StopIteration:
            pass

    layers[0] = catcher._layer
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    model.config.use_cache = use_cache

    # Move all kwargs tensors to dev
    layer_kwargs = cache["kwargs"] or {}
    device_kwargs: Dict[str, Any] = {}
    for k, v in layer_kwargs.items():
        if isinstance(v, torch.Tensor):
            device_kwargs[k] = v.to(dev)
        elif isinstance(v, (list, tuple)):
            device_kwargs[k] = type(v)(
                x.to(dev) if isinstance(x, torch.Tensor) else x
                for x in v
            )
        else:
            device_kwargs[k] = v

    return inps, device_kwargs


class Qwen35LayerReplay:
    """
    Manages layer-by-layer forward pass for Qwen3.5, correctly propagating
    both the attention KV-cache (for full-attention layers) and the recurrent
    state (for DeltaNet layers).

    Usage:
        replay = Qwen35LayerReplay(model, input_ids, dev)
        for layer_idx in range(num_layers):
            replay.set_layer(layer_idx)
            for sample_idx in range(nsamples):
                out = replay.forward_sample(sample_idx)   # runs the layer
            replay.commit()   # swap inps/outs, advance
    """

    def __init__(self, model, input_ids: torch.Tensor, dev: str):
        self.model = model
        self.dev = dev
        self.layers = model.model.layers
        self.layer_types = get_layer_types(model)
        self.nsamples = int(input_ids.size(0))

        # Collect initial inputs
        self.inps, self.layer_kwargs = collect_qwen35_calibration_inputs(
            model, input_ids, dev
        )
        self.outs = torch.zeros_like(self.inps)

        # Per-sample recurrent states for DeltaNet layers.
        # Shape: [nsamples] of (Tensor or None)
        self.recurrent_states: List[Optional[torch.Tensor]] = [None] * self.nsamples

        self._current_layer: Optional[nn.Module] = None
        self._current_idx: int = -1

    def set_layer(self, layer_idx: int) -> nn.Module:
        """Move layer to GPU, return it."""
        layer = self.layers[layer_idx].to(self.dev)
        self._current_layer = layer
        self._current_idx = layer_idx
        return layer

    @torch.no_grad()
    def forward_sample(self, sample_idx: int) -> torch.Tensor:
        """
        Run current layer on sample `sample_idx`.
        Returns the output hidden states [1, seqlen, hidden].
        """
        layer = self._current_layer
        inp = self.inps[sample_idx:sample_idx + 1].to(self.dev)
        kwargs = self._build_sample_kwargs(sample_idx)

        out = layer(inp, **kwargs)

        # Qwen3.5 decoder layers return a tuple.
        # Layout depends on layer type:
        #   full-attention:  (hidden_states, attn_weights?, recurrent_state?)
        #   linear-attention: (hidden_states, recurrent_state)
        # We always take [0] as the hidden states and try to extract recurrent state.
        if isinstance(out, (tuple, list)):
            hidden_out = out[0]
            # Try to find and save new recurrent state for DeltaNet layers
            self._update_recurrent_state(sample_idx, out)
        else:
            hidden_out = out

        self.outs[sample_idx].copy_(hidden_out.squeeze(0).detach())
        return hidden_out

    def commit(self) -> None:
        """Swap inps/outs after all samples have been processed for current layer."""
        layer = self._current_layer
        if layer is not None:
            self.layers[self._current_idx] = layer.cpu()
        self._current_layer = None
        gc.collect()
        torch.cuda.empty_cache()
        self.inps, self.outs = self.outs, self.inps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sample_kwargs(self, sample_idx: int) -> Dict[str, Any]:
        """Build per-sample kwargs dict, injecting the correct recurrent_state."""
        kwargs: Dict[str, Any] = {}
        for k, v in self.layer_kwargs.items():
            if k == "recurrent_state":
                continue  # handled separately
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(self.dev)
            elif isinstance(v, (list, tuple)):
                kwargs[k] = type(v)(
                    x.to(self.dev) if isinstance(x, torch.Tensor) else x
                    for x in v
                )
            else:
                kwargs[k] = v

        # Inject per-sample recurrent state
        rs = self.recurrent_states[sample_idx]
        if rs is not None:
            kwargs["recurrent_state"] = rs.to(self.dev)
        else:
            kwargs["recurrent_state"] = None

        return kwargs

    def _update_recurrent_state(self, sample_idx: int, out) -> None:
        """
        Extract and save the new recurrent state for DeltaNet layers.

        Qwen3.5 DeltaNet forward returns a tuple where the last non-None
        tensor of appropriate rank is the recurrent state
        [bs, num_v_heads, k_head_dim, v_head_dim].
        """
        if not is_deltanet_layer(self.layer_types, self._current_idx):
            return  # full-attn layers don't have persistent recurrent state

        if not isinstance(out, (tuple, list)):
            return

        # Find last tensor in the output tuple that is 4-D (the recurrent state)
        new_state = None
        for item in reversed(out):
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                new_state = item
                break

        if new_state is not None:
            self.recurrent_states[sample_idx] = new_state.detach().cpu()


# ---------------------------------------------------------------------------
# GPTQ sequential groups for Qwen3.5
# ---------------------------------------------------------------------------

#: Groups of projections that share the same Hessian capture window.
#: This matches TWLA's Llama convention but adapted to Qwen3.5 module names.
FULL_ATTN_SEQUENTIAL: List[List[str]] = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]

DELTANET_SEQUENTIAL: List[List[str]] = [
    ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
     "linear_attn.in_proj_b", "linear_attn.in_proj_a"],
    ["linear_attn.out_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


def get_sequential_for_layer(layer_types: List[str], layer_idx: int) -> List[List[str]]:
    """Return the appropriate sequential quantization groups for this layer."""
    if is_deltanet_layer(layer_types, layer_idx):
        return DELTANET_SEQUENTIAL
    return FULL_ATTN_SEQUENTIAL


# ---------------------------------------------------------------------------
# Main forward pass for GPTQ-style quantization (replaces twla_fwrd for Qwen3.5)
# ---------------------------------------------------------------------------

def twla_fwrd_qwen35(model, tokenizer, dataloader, dev, args):
    """
    Qwen3.5-aware replacement for gptq_fwrd.twla_fwrd().

    Correctly:
      - Uses Qwen35LayerReplay to propagate recurrent state across DeltaNet layers
      - Uses model-appropriate sequential groups per layer
      - Reuses the existing Ternarization + TRAGPTQ quantization kernels unchanged
    """
    from quantize.E2M_ATQ import Ternarization
    from quantize.tra_gptq import TRAGPTQ
    from quantize.gptq_fwrd import find_qlayers

    print("-----E2M-ATQ Quantization (Qwen3.5)-----")

    if args.group_size is None:
        args.group_size = -1

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Collect calibration data
    all_input_ids = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        else:
            input_ids = batch["input_ids"]
        all_input_ids.append(input_ids)
        if len(all_input_ids) >= args.nsamples:
            break

    # Stack into [nsamples, seqlen]
    input_ids = torch.cat(all_input_ids, dim=0)[:args.nsamples]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0).expand(args.nsamples, -1)

    replay = Qwen35LayerReplay(model, input_ids, dev)
    layer_types = replay.layer_types
    layers = replay.layers

    quantizers = {}
    print(f"Total layers: {len(layers)}")

    for i in range(len(layers)):
        print(f"\nLayer {i} ({layer_types[i]}):", flush=True, end=" ")
        layer = replay.set_layer(i)

        full = find_qlayers(layer, layers=[torch.nn.Linear])
        sequential = get_sequential_for_layer(layer_types, i)

        for names in sequential:
            # Filter to only those that actually exist in this layer
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            gptq: Dict[str, Any] = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                if "lm_head" in name:
                    continue
                braq_quantizer = Ternarization(
                    subset[name].weight,
                    groupsize=128,
                )
                gptq[name] = TRAGPTQ(
                    layer=subset[name],
                    braq_quantizer=braq_quantizer,
                    salient_metric=getattr(args, "salient_metric", "hessian"),
                    disable_gptq=getattr(args, "disable_gptq_core", False),
                    order2_group=getattr(args, "order2_group", True),
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(replay.nsamples):
                replay.forward_sample(j)

            for h in handles:
                h.remove()

            for name in gptq:
                logging.info(f"Quantizing {i}/{name}")
                gptq[name].fasterquant(
                    blocksize=getattr(args, "blocksize", 128),
                    percdamp=getattr(args, "percdamp", 0.1),
                    orders=(1, 1, 2),
                    num_p=getattr(args, "num_p", 1),
                )
                quantizers[f"model.{replay._current_idx}.{name}"] = gptq[name]
                gptq[name].free()

        # Re-run all samples through the now-quantized layer to get correct outs
        for j in range(replay.nsamples):
            replay.forward_sample(j)

        replay.commit()

    # Move lm_head to GPU (same as original twla_fwrd)
    print("\nhead:", flush=True, end=" ")
    model.lm_head = model.lm_head.to(device=dev, dtype=torch.float32)
    print("head finished")

    model.config.use_cache = use_cache
    gc.collect()
    torch.cuda.empty_cache()
    print("-----E2M-ATQ Quantization Done (Qwen3.5)-----\n")
    return quantizers


# ---------------------------------------------------------------------------
# KOTMS export helper — replaces the generic model.model.named_modules() walk
# ---------------------------------------------------------------------------

def get_kotms_targets(model) -> List[Tuple[str, nn.Linear]]:
    """
    Return (name, module) pairs for KOTMS rotation.
    These are exactly the projections TWLA should ternarize.
    """
    return get_kotms_linear_layers(model)


# ---------------------------------------------------------------------------
# Export checkpoint helpers
# ---------------------------------------------------------------------------

def export_rotated_checkpoint_qwen35(model: nn.Module, save_path: str, meta: Dict) -> None:
    """
    Export KOTMS-rotated checkpoint using the Qwen3.5-aware module list.
    Mirrors scripts/KOTMS.py::export_rotated_checkpoint() but scoped to
    Qwen3.5 target modules only.
    """
    import os
    pkg: Dict[str, Any] = {"_meta": dict(meta), "layers": {}}
    targets = get_kotms_linear_layers(model)
    for full_name, layer in targets:
        d: Dict[str, Any] = {
            "weight": layer.weight.detach().cpu().half().contiguous(),
            "bias": (None if layer.bias is None
                     else layer.bias.detach().cpu().half().contiguous()),
            "L": None,
            "R": None,
            "dim_l": int(getattr(layer, "dim_l", 0) or 0),
            "dim_r": int(getattr(layer, "dim_r", 0) or 0),
        }
        for k in ("L", "R"):
            t = getattr(layer, k, None)
            d[k] = (None if (t is None or not isinstance(t, torch.Tensor))
                    else t.detach().cpu().half().contiguous())
        pkg["layers"][full_name] = d

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(pkg, save_path)


def import_rotated_checkpoint_qwen35(model, load_path: str, args) -> int:
    """
    Load KOTMS-rotated weights into a Qwen3.5 model.
    Returns the number of layers filled.
    """
    # Import QLinear from run_twla_qwen35 (no circular dependency as that
    # imports from us but not vice versa at module load time).
    # Alternatively, define a minimal QLinear locally.
    try:
        from run_twla_qwen35 import QLinear
    except ImportError:
        # Fallback: minimal QLinear with no activation quantization
        class QLinear(nn.Linear):  # type: ignore[no-redef]
            def init_act_quant(self, args) -> None:
                self.act_quant = None
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                init_shape = x.shape
                if hasattr(self, "L") and self.L is not None:
                    device = x.device
                    L = self.L.to(device) if self.L.device != device else self.L
                    R = self.R.to(device) if self.R.device != device else self.R
                    x = x.reshape(-1, self.dim_l, self.dim_r)
                    x = L @ x @ R
                    x = x.reshape(init_shape)
                if getattr(self, "act_quant", None) is not None:
                    x = self.act_quant(x)
                return super().forward(x)

    pkg = torch.load(load_path, map_location="cpu")
    layers_pkg = pkg["layers"]
    filled = 0

    targets = get_kotms_linear_layers(model)
    target_dict = {name: mod for name, mod in targets}

    for name, d in layers_pkg.items():
        if name not in target_dict:
            continue
        layer = target_dict[name]

        layer.weight.data = d["weight"].to(dtype=torch.float16)
        if d["bias"] is not None:
            if layer.bias is None:
                layer.bias = nn.Parameter(d["bias"].to(dtype=torch.float16))
            else:
                layer.bias.data = d["bias"].to(dtype=torch.float16)

        layer.__class__ = QLinear

        def _reg(k: str) -> None:
            t = d.get(k, None)
            if t is None:
                return
            t = t.to(dtype=torch.float16).contiguous()
            if hasattr(layer, k) and isinstance(getattr(layer, k), torch.Tensor):
                getattr(layer, k).data.copy_(t)
            else:
                layer.register_buffer(k, t, persistent=True)

        _reg("L")
        _reg("R")
        if int(d.get("dim_l", 0) or 0) > 0:
            layer.dim_l = int(d["dim_l"])
        if int(d.get("dim_r", 0) or 0) > 0:
            layer.dim_r = int(d["dim_r"])

        if hasattr(layer, "init_act_quant"):
            layer.init_act_quant(args)
        filled += 1

    return filled

