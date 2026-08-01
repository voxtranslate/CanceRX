import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path

# =============================================
# MODULAR MODEL COMPONENTS
# =============================================

class CMPA(nn.Module):
    """Cosine-Modulated Phase Attention"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.phase_net = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)

        # Zero-initialize the Phase Net. 
        # This keeps the gate fully "open" at step 1 so gradients aren't chaotically blocked.
        nn.init.constant_(self.phase_net.weight, 0.0)
        nn.init.constant_(self.phase_net.bias, 0.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        phases = self.phase_net(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        phase_diff = phases.mean(dim=-1, keepdim=True) - phases.mean(dim=-1, keepdim=True).transpose(-2, -1)
        interference = torch.cos(phase_diff) 
        
        attn = attn * interference 
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out), attn


class TPR(nn.Module):
    """TopologicalPathwayRouting"""
    def __init__(self, dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(nn.Linear(dim, num_experts), nn.Softmax(dim=-1))

        nn.init.constant_(self.router[0].weight, 0.0)
        nn.init.constant_(self.router[0].bias, 0.0)

    def forward(self, x):
        routes = self.router(x)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + routes[:, :, i:i + 1] * expert(x)
        return out, routes


class TPGBlock(nn.Module):
    """Topological Phased Gated Block"""
    def __init__(self, dim, num_heads, num_experts):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CMPA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.tpr = TPR(dim, num_experts)

    def forward(self, x):
        attn_out, attn_map = self.attn(self.norm1(x))
        x = x + attn_out
        tpr_out, route_map = self.tpr(self.norm2(x))
        x = x + tpr_out
        return x, attn_map, route_map


class CanceRX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.img_size = config.img_size
        self.grid = config.img_size // self.patch_size
        self.patch_embed = nn.Conv2d(
            config.in_chans, config.dim,
            kernel_size=self.patch_size, stride=self.patch_size
        )
        self.num_patches = self.grid ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, config.dim) * 0.02)

        self.layers = nn.ModuleList([
            TPGBlock(config.dim, config.num_heads, config.num_experts)
            for _ in range(config.depth)
        ])

        self.norm = nn.LayerNorm(config.dim)
        self.head_type = nn.Linear(config.dim, config.num_types)
        self.head_stage = nn.Linear(config.dim, config.num_stages)

        self._cam_activations = None
        self._cam_gradients = None
        self._cam_handles = []

    def _fwd_hook(self, module, inp, out):
        self._cam_activations = out
        if out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._cam_gradients = grad

    def enable_cam_hooks(self):
        self.disable_cam_hooks()
        h = self.patch_embed.register_forward_hook(self._fwd_hook)
        self._cam_handles.append(h)

    def disable_cam_hooks(self):
        for h in self._cam_handles:
            h.remove()
        self._cam_handles = []
        self._cam_activations = None
        self._cam_gradients = None

    @property
    def cam_activations(self):
        return self._cam_activations

    @property
    def cam_gradients(self):
        return self._cam_gradients

    def forward(self, x, return_internals=False):
        feat_map = self.patch_embed(x)                      
        x = feat_map.flatten(2).transpose(1, 2)              
        x = x + self.pos_embed

        attns, routes = [], []
        for layer in self.layers:
            x, attn_map, route_map = layer(x)
            if return_internals:
                attns.append(attn_map)
                routes.append(route_map)

        feats = self.norm(x.mean(dim=1))

        logits_type = self.head_type(feats)
        logits_stage = self.head_stage(feats)

        if return_internals:
            return logits_type, logits_stage, attns, routes, feats
        return logits_type, logits_stage, None, None, feats