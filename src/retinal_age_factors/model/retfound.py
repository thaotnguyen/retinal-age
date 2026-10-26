"""RETFound DINOv2 (ViT-L/14) with a single-output age-regression head.

``param_groups_lrd`` and ``interpolate_pos_embed`` are adapted from the
RETFound repository (https://github.com/rmaphoh/RETFound, util/lr_decay.py and
util/pos_embed.py; Copyright (c) Meta Platforms, Inc. and affiliates, partly
revised by YZ @UCL&Moorfields; CC BY-NC 4.0), which follow BEiT and MAE.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

LOG = logging.getLogger(__name__)

CHECKPOINT_REPO = "YukunZhou/RETFound_dinov2_meh"
CHECKPOINT_FILE = "RETFound_dinov2_meh.pth"
TIMM_ARCHITECTURE = "vit_large_patch14_dinov2.lvd142m"


def _layer_id(name: str, num_layers: int) -> int:
    if name in {"cls_token", "pos_embed"} or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    return num_layers


def param_groups_lrd(model: nn.Module, weight_decay: float, layer_decay: float) -> list[dict]:
    """Parameter groups with layer-wise learning-rate decay (BEiT recipe)."""
    num_layers = len(model.blocks) + 1
    scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    no_decay = model.no_weight_decay()
    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay = 0.0 if param.ndim == 1 or name in no_decay else weight_decay
        layer = _layer_id(name, num_layers)
        key = f"layer_{layer}_{'no_decay' if decay == 0.0 else 'decay'}"
        group = groups.setdefault(key, {"lr_scale": scales[layer], "weight_decay": decay, "params": []})
        group["params"].append(param)
    return list(groups.values())


def interpolate_pos_embed(model: nn.Module, state: dict) -> None:
    """Resize a checkpoint's position embedding to the model's patch grid."""
    if "pos_embed" not in state:
        return
    checkpoint = state["pos_embed"]
    num_patches = model.patch_embed.num_patches
    extra = model.pos_embed.shape[-2] - num_patches
    old_size = int((checkpoint.shape[-2] - extra) ** 0.5)
    new_size = int(num_patches**0.5)
    if old_size == new_size:
        return
    LOG.info("Position interpolate from %dx%d to %dx%d", old_size, old_size, new_size, new_size)
    tokens = checkpoint[:, extra:].reshape(-1, old_size, old_size, checkpoint.shape[-1]).permute(0, 3, 1, 2)
    tokens = nn.functional.interpolate(tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    state["pos_embed"] = torch.cat((checkpoint[:, :extra], tokens.permute(0, 2, 3, 1).flatten(1, 2)), dim=1)


def build_model(image_size: int, drop_path: float, pretrained: bool) -> nn.Module:
    """ViT-L/14 DINOv2 with RETFound DINOv2-MEH weights and a fresh age head."""
    model = timm.create_model(
        TIMM_ARCHITECTURE,
        pretrained=pretrained,
        img_size=image_size,
        num_classes=1,
        drop_path_rate=drop_path,
    )
    if not pretrained:
        return model
    checkpoint = torch.load(hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE), map_location="cpu", weights_only=False)
    state = {
        key.replace("backbone.", "").replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2."): value
        for key, value in checkpoint["teacher"].items()
    }
    for key in ("head.weight", "head.bias"):
        if key in state and state[key].shape != model.state_dict()[key].shape:
            del state[key]
    interpolate_pos_embed(model, state)
    message = model.load_state_dict(state, strict=False)
    unexpected = [k for k in message.unexpected_keys if not k.startswith(("mask_token", "dino_head.", "ibot_head."))]
    if unexpected:
        raise RuntimeError(f"Unexpected RETFound checkpoint keys: {unexpected}")
    nn.init.trunc_normal_(model.head.weight, std=2e-5)
    nn.init.zeros_(model.head.bias)
    return model


def checkpoint_sha256() -> str:
    digest = hashlib.sha256()
    with Path(hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
