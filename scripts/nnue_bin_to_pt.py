"""Recover a PyTorch .pt checkpoint from a quantized .bin weight file.

The .bin stores int16/int8 quantized weights. We dequantize them back to
float32 to enable resuming training. The dequantization is approximate
(the original float scale is not stored), but close enough for fine-tuning.
"""

import numpy as np
import torch
import torch.nn as nn
import struct
import sys
import os

FT_N = 256
L1_N = 32
NUM_FEATURES = 64 * 64 * 12


class NNUE(nn.Module):
    """Same architecture as the training scripts."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(NUM_FEATURES, FT_N, mode='sum')
        self.ft_bias = nn.Parameter(torch.zeros(FT_N))
        self.l1 = nn.Linear(FT_N * 2, L1_N, bias=True)
        self.l2 = nn.Linear(L1_N, 1, bias=True)

    def forward(self, wk_features, bk_features):
        batch_size = len(wk_features)
        device = self.ft_bias.device

        wk_lens = torch.tensor([f.size(0) for f in wk_features], device=device)
        wk_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        wk_offsets[1:] = wk_lens.cumsum(dim=0)
        wk_indices = torch.cat([f.to(device) for f in wk_features])

        bk_lens = torch.tensor([f.size(0) for f in bk_features], device=device)
        bk_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        bk_offsets[1:] = bk_lens.cumsum(dim=0)
        bk_indices = torch.cat([f.to(device) for f in bk_features])

        acc_w = self.embedding(wk_indices, wk_offsets[:-1]) + self.ft_bias.unsqueeze(0)
        acc_b = self.embedding(bk_indices, bk_offsets[:-1]) + self.ft_bias.unsqueeze(0)
        clipped_w = torch.clamp(acc_w, 0, 255)
        clipped_b = torch.clamp(acc_b, 0, 255)
        h = torch.cat([clipped_w, clipped_b], dim=1)
        h = self.l1(h)
        h = torch.clamp(h, 0, 127)
        out = self.l2(h)
        return out.squeeze(-1)


def load_bin(path: str) -> dict:
    """Load quantized weights from .bin file, return state_dict."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'NNUE':
            raise ValueError(f"Bad magic: {magic}")

        version = struct.unpack('<i', f.read(4))[0]
        ft_n = struct.unpack('<i', f.read(4))[0]
        l1_n = struct.unpack('<i', f.read(4))[0]

        print(f"  version={version}, ft_n={ft_n}, l1_n={l1_n}")

        if ft_n != FT_N or l1_n != L1_N:
            raise ValueError(f"Architecture mismatch: FT_N={ft_n}, L1_N={l1_n}")

        ft_bias_q = np.frombuffer(f.read(FT_N * 2), dtype=np.int16).copy()
        print(f"  ft_bias: [{ft_bias_q.min()}, {ft_bias_q.max()}]")

        ft_weight_q = np.frombuffer(f.read(NUM_FEATURES * FT_N * 2), dtype=np.int16).copy()
        ft_weight_q = ft_weight_q.reshape(NUM_FEATURES, FT_N)
        print(f"  ft_weight: [{ft_weight_q.min()}, {ft_weight_q.max()}]")

        l1_weight_q = np.frombuffer(f.read(FT_N * 2 * L1_N), dtype=np.int8).copy()
        l1_weight_q = l1_weight_q.reshape(FT_N * 2, L1_N)
        print(f"  l1_weight: [{l1_weight_q.min()}, {l1_weight_q.max()}]")

        l1_bias_q = np.frombuffer(f.read(L1_N), dtype=np.int8).copy()
        print(f"  l1_bias: [{l1_bias_q.min()}, {l1_bias_q.max()}]")

        l2_weights_q = np.frombuffer(f.read(L1_N), dtype=np.int8).copy()
        print(f"  l2_weights: [{l2_weights_q.min()}, {l2_weights_q.max()}]")

        l2_bias_q = struct.unpack('<h', f.read(2))[0]
        print(f"  l2_bias: {l2_bias_q}")

    # Estimate dequantization scales.
    # The quantizer maps max(abs(x)) -> 8191 (or 63 for int8).
    # We back-estimate so that max(abs(float)) ≈ 1.0 for FT layers.
    ft_scale = max(np.max(np.abs(ft_weight_q)), np.max(np.abs(ft_bias_q)), 1.0)
    ft_scale = ft_scale / 1.0  # target max abs float = 1.0

    l1_scale = max(np.max(np.abs(l1_weight_q)), np.max(np.abs(l1_bias_q)), 1.0)
    l1_scale = l1_scale / 1.0

    l2_scale = max(np.max(np.abs(l2_weights_q)), 1.0)
    l2_scale = l2_scale / 1.0

    print(f"  ft_scale={ft_scale:.4f}, l1_scale={l1_scale:.4f}, l2_scale={l2_scale:.4f}")

    ft_weight = ft_weight_q.astype(np.float32) / ft_scale
    ft_bias = ft_bias_q.astype(np.float32) / ft_scale

    l1_weight = l1_weight_q.astype(np.float32) / l1_scale
    l1_bias = l1_bias_q.astype(np.float32) / l1_scale

    l2_weights = l2_weights_q.astype(np.float32) / l2_scale
    l2_bias = float(l2_bias_q) / ft_scale  # same scale as ft

    print(f"  Dequantized ft_weight range: [{ft_weight.min():.4f}, {ft_weight.max():.4f}]")
    print(f"  Dequantized ft_bias range: [{ft_bias.min():.4f}, {ft_bias.max():.4f}]")

    # Build state_dict
    # nnue_train_gpu.py uses EmbeddingBag, so ft_weight maps to embedding.weight
    state = {
        'embedding.weight': torch.from_numpy(ft_weight),
        'ft_bias': torch.from_numpy(ft_bias),
        'l1.weight': torch.from_numpy(l1_weight.T),  # stored as (FT*2, L1), Linear expects (L1, FT*2)
        'l1.bias': torch.from_numpy(l1_bias),
        'l2.weight': torch.from_numpy(l2_weights.reshape(1, L1_N)),  # Linear expects (1, L1_N)
        'l2.bias': torch.tensor([l2_bias], dtype=torch.float32),
    }

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="nnue.bin", help="Quantized .bin file")
    parser.add_argument("--output", type=str, default="", help="Output .pt file (default: input .bin stem + .pt)")
    args = parser.parse_args()

    if not args.output:
        args.output = args.input.replace('.bin', '.pt') if args.input.endswith('.bin') else args.input + '.pt'

    print(f"Reading {args.input}...")
    state = load_bin(args.input)

    model = NNUE()
    model.load_state_dict(state)

    torch.save(model.state_dict(), args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    import argparse
    main()
