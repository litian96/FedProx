from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class FedProxOptimizer(Optimizer):
    """PyTorch implementation of the FedProx Perturbed Gradient Descent optimizer.

    Solves the local client optimization subproblem with proximal regularization:
        min_w F_k(w) + (mu / 2) * ||w - w*||^2

    Parameter update:
        w <- w - lr * (g + mu * (w - w*))
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.01,
        mu: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid proximal coefficient mu: {mu}")

        defaults = {
            "lr": lr,
            "mu": mu,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        # Initialize slots for global reference weights w*
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["w_star"] = p.detach().clone()
                if momentum > 0.0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

    def set_global_weights(
        self, global_weights: nn.Module | dict[str, torch.Tensor] | list[torch.Tensor]
    ):
        """Update the global reference weights w* at the start of a local round."""
        if isinstance(global_weights, nn.Module):
            ref_params = list(global_weights.parameters())
        elif isinstance(global_weights, dict):
            ref_params = list(global_weights.values())
        else:
            ref_params = list(global_weights)

        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if idx < len(ref_params):
                    self.state[p]["w_star"] = (
                        ref_params[idx].detach().clone().to(p.device)
                    )
                    idx += 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                state = self.state[p]

                # Weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                # Proximal regularization term: mu * (w - w*)
                if mu > 0 and "w_star" in state:
                    d_p = d_p.add(p.data - state["w_star"], alpha=mu)

                # Momentum
                if momentum > 0:
                    buf = state.get("momentum_buffer", None)
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-lr)

        return loss


def quantize_tensor(
    tensor: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    residual: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    tuple[int, ...],
    torch.Tensor | None,
]:
    """Block-wise uniform symmetric quantization with sub-byte bit packing (torch.uint8)."""
    orig_shape = tensor.shape
    orig_device = tensor.device
    orig_dtype = tensor.dtype

    work = tensor.detach().clone().to(torch.float32)
    if residual is not None and residual.shape == orig_shape:
        work = work + residual.to(device=orig_device, dtype=torch.float32)

    flat = work.reshape(-1)
    numel = flat.numel()

    if numel == 0:
        empty_packed = torch.empty((0,), dtype=torch.uint8, device=orig_device)
        empty_scales = torch.empty((0,), dtype=torch.float32, device=orig_device)
        empty_res = torch.zeros(orig_shape, dtype=orig_dtype, device=orig_device)
        return empty_packed, empty_scales, None, orig_shape, empty_res

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    blocks = padded.view(-1, block_size)
    qmax = (1 << (nbits - 1)) - 1  # 7 for 4-bit
    qmin = -(1 << (nbits - 1))  # -8 for 4-bit

    max_val = torch.max(torch.abs(blocks), dim=1, keepdim=True)[0]
    scales = torch.clamp(max_val / float(qmax), min=1e-8)

    scaled_blocks = blocks / scales
    quantized_blocks = torch.clamp(torch.round(scaled_blocks), qmin, qmax)

    dequantized_padded = (quantized_blocks * scales).view(-1)
    dequantized_flat = dequantized_padded[:numel]
    new_residual = (flat - dequantized_flat).view(orig_shape).to(orig_dtype)

    unsigned_blocks = (quantized_blocks - qmin).to(torch.uint8)
    unsigned_flat = unsigned_blocks.view(-1)

    if nbits == 4:
        even_nibbles = unsigned_flat[0::2]
        odd_nibbles = unsigned_flat[1::2]
        packed = (even_nibbles & 0x0F) | ((odd_nibbles & 0x0F) << 4)
    elif nbits == 8:
        packed = unsigned_flat
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    return packed, scales.squeeze(-1), None, orig_shape, new_residual


def dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    orig_shape: tuple[int, ...] | None = None,
    nbits: int = 4,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed integers back into float representation."""
    if packed.numel() == 0:
        if orig_shape is not None:
            return torch.zeros(orig_shape, dtype=dtype, device=packed.device)
        return torch.empty((0,), dtype=dtype, device=packed.device)

    qmin = -(1 << (nbits - 1))

    if nbits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unsigned = torch.stack([low, high], dim=-1).view(-1)
    elif nbits == 8:
        unsigned = packed
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    signed = unsigned.to(torch.float32) + float(qmin)
    blocks = signed.view(-1, block_size)
    recon_blocks = blocks * scales.view(-1, 1)
    recon_flat = recon_blocks.view(-1)

    if orig_shape is not None:
        numel = 1
        for dim in orig_shape:
            numel *= dim
        recon_flat = recon_flat[:numel]
        return recon_flat.view(orig_shape).to(dtype)

    return recon_flat.to(dtype)


def quantize_model_delta(
    local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
    residuals: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, float]]:
    """Quantize parameter differences delta = local - global with error feedback."""
    quantized_dict: dict[str, Any] = {
        "nbits": nbits,
        "block_size": block_size,
        "layers": {},
    }
    new_residuals: dict[str, torch.Tensor] = {}

    total_orig_bytes = 0
    total_quant_bytes = 0

    if residuals is None:
        residuals = {}

    for name, local_p in local_state.items():
        global_p = global_state[name]
        delta = local_p - global_p

        orig_bytes = delta.numel() * delta.element_size()
        total_orig_bytes += orig_bytes

        if not delta.is_floating_point():
            quantized_dict["layers"][name] = {
                "type": "passthrough",
                "tensor": delta.detach().clone(),
            }
            total_quant_bytes += orig_bytes
            continue

        res = residuals.get(name, None)
        packed, scales, zeros, orig_shape, new_res = quantize_tensor(
            delta, nbits=nbits, block_size=block_size, residual=res
        )

        quant_bytes = (
            packed.numel() * packed.element_size()
            + scales.numel() * scales.element_size()
        )
        total_quant_bytes += quant_bytes

        quantized_dict["layers"][name] = {
            "type": "quantized",
            "packed": packed,
            "scales": scales,
            "zeros": zeros,
            "orig_shape": orig_shape,
            "dtype": str(delta.dtype),
        }
        new_residuals[name] = new_res

    compression_ratio = float(total_orig_bytes) / max(total_quant_bytes, 1)
    stats = {
        "orig_bytes": total_orig_bytes,
        "quant_bytes": total_quant_bytes,
        "compression_ratio": compression_ratio,
        "bandwidth_savings_pct": (1.0 - (total_quant_bytes / max(total_orig_bytes, 1)))
        * 100.0,
    }

    return quantized_dict, new_residuals, stats


def dequantize_model_delta(
    quantized_dict: dict[str, Any],
    device: torch.device | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Dequantize client parameter differences for server aggregation."""
    nbits = quantized_dict.get("nbits", 4)
    block_size = quantized_dict.get("block_size", 64)
    layers = quantized_dict.get("layers", {})

    dequantized_delta: OrderedDict[str, torch.Tensor] = OrderedDict()

    for name, data in layers.items():
        if data.get("type") == "passthrough":
            tensor = data["tensor"]
            if device is not None:
                tensor = tensor.to(device)
            dequantized_delta[name] = tensor
        else:
            packed = data["packed"]
            scales = data["scales"]
            zeros = data.get("zeros")
            orig_shape = data["orig_shape"]

            recon = dequantize_tensor(
                packed=packed,
                scales=scales,
                zeros=zeros,
                orig_shape=orig_shape,
                nbits=nbits,
                block_size=block_size,
            )
            if device is not None:
                recon = recon.to(device)
            dequantized_delta[name] = recon

    return dequantized_delta
