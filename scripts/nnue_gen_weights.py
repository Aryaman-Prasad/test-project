"""
NNUE weight generation: random FT + trained L1/L2 via sklearn.

Strategy:
  - Feature Transformer: random int16 weights (fixed seed)
  - L1/L2: trained to map FT output -> HCE score via MLPRegressor
  - Export to C++ binary format

This avoids end-to-end GPU training while producing reasonable weights.
"""

import numpy as np
import struct
import random
import os
import time
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_EXPORT = os.path.join(PROJECT_ROOT, "nnue.bin")

FT_N = 256
L1_N = 32
NUM_FEATURES = 64 * 64 * 12

# Piece lookup
PIECE_MAP = {
    'P': (0, 0), 'N': (0, 1), 'B': (0, 2), 'R': (0, 3), 'Q': (0, 4), 'K': (0, 5),
    'p': (1, 0), 'n': (1, 1), 'b': (1, 2), 'r': (1, 3), 'q': (1, 4), 'k': (1, 5),
}


def file_of(sq): return sq & 7
def rank_of(sq): return sq >> 3

def square_distance(a, b):
    fd = abs(file_of(a) - file_of(b))
    rd = abs(rank_of(a) - rank_of(b))
    return fd if fd > rd else rd


def make_feature_index(king_sq, piece_sq, piece_idx):
    return (king_sq * 64 + piece_sq) * 12 + piece_idx


def active_features(board):
    wk_sq = board.index('K')
    bk_sq = board.index('k')
    wk_f, bk_f = [], []
    for sq, pc in enumerate(board):
        if pc == '.': continue
        color, pt = PIECE_MAP[pc]
        pidx = color * 6 + pt
        if not (color == 0 and pt == 5):
            wk_f.append(make_feature_index(wk_sq, sq, pidx))
        if not (color == 1 and pt == 5):
            bk_f.append(make_feature_index(bk_sq, sq, pidx))
    return wk_f, bk_f


def random_fen():
    board = ['.'] * 64
    kw = random.randint(0, 63)
    kb = random.randint(0, 63)
    while square_distance(kw, kb) < 2 or file_of(kw) == file_of(kb):
        kb = random.randint(0, 63)
    board[kw] = 'K'
    board[kb] = 'k'

    for piece, max_count in [
        ('P', 8), ('p', 8), ('N', 2), ('n', 2),
        ('B', 2), ('b', 2), ('R', 2), ('r', 2),
        ('Q', 1), ('q', 1)
    ]:
        count = random.randint(0, max_count)
        for _ in range(count):
            empty = [i for i, c in enumerate(board) if c == '.']
            if not empty: break
            sq = random.choice(empty)
            color, pt_idx = PIECE_MAP[piece]
            if pt_idx == 0 and (sq >> 3) in (0, 7): continue
            board[sq] = piece

    rows = []
    for r in range(7, -1, -1):
        row = ''
        empty_run = 0
        for f in range(8):
            c = board[r * 8 + f]
            if c == '.':
                empty_run += 1
            else:
                if empty_run: row += str(empty_run); empty_run = 0
                row += c
        if empty_run: row += str(empty_run)
        rows.append(row)
    board_str = '/'.join(rows)
    stm = 'w' if random.random() < 0.5 else 'b'
    return f"{board_str} {stm} KQkq - 0 1"


# --- HCE ---
def relative_sq(sq, color):
    return sq if color == 0 else sq ^ 56

HCE_VALUES = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0}
HCE_PST = {
    'P': [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0],
    'N': [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50],
    'B': [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20],
    'R': [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0],
    'Q': [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20],
}
HCE_KING_MID = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]
HCE_KING_END = [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,0,0,-10,-20,-30,-30,-10,20,30,30,20,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,20,30,30,20,-10,-30,-30,-30,0,0,0,0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50]


def hce_evaluate(fen):
    parts = fen.split()
    rows = parts[0].split('/')
    board = ['.'] * 64
    for ri, row in enumerate(rows):
        fi = 0
        for c in row:
            if c.isdigit(): fi += int(c)
            else: board[ri * 8 + fi] = c; fi += 1
    stm = 0 if parts[1] == 'w' else 1

    mat = [0, 0]; mg = [0, 0]; eg = [0, 0]
    for sq, pc in enumerate(board):
        if pc == '.': continue
        c = 0 if pc.isupper() else 1
        mat[c] += HCE_VALUES.get(pc.upper(), 0)
        pst = HCE_PST.get(pc)
        if pst: mg[c] += pst[relative_sq(sq, c)]

    for c in [0, 1]:
        kc = 'K' if c == 0 else 'k'
        for sq, pc in enumerate(board):
            if pc == kc:
                rsq = relative_sq(sq, c)
                mg[c] += HCE_KING_MID[rsq]
                eg[c] += HCE_KING_END[rsq]
                break

    for c in [0, 1]:
        pc = 'P' if c == 0 else 'p'
        pawns = [sq for sq, cp in enumerate(board) if cp == pc]
        for sq in pawns:
            f = sq & 7
            if sum(1 for psq in pawns if (psq & 7) == f) > 1: mg[c] -= 15
            has_adj = False
            for psq in pawns:
                pf = psq & 7
                if pf in (f - 1, f + 1):
                    if (c == 0 and (psq >> 3) >= (sq >> 3)) or \
                       (c == 1 and (psq >> 3) <= (sq >> 3)):
                        has_adj = True
            if not has_adj: mg[c] -= 10

    for c in [0, 1]:
        bc = 'B' if c == 0 else 'b'
        if sum(1 for cp in board if cp == bc) >= 2: mat[c] += 30

    for c in [0, 1]:
        rc = 'R' if c == 0 else 'r'; pc = 'P' if c == 0 else 'p'
        for sq, cp in enumerate(board):
            if cp == rc:
                if not any((psq & 7) == (sq & 7) for psq, cp2 in enumerate(board) if cp2 == pc):
                    mg[c] += 20

    phase = sum(
        1 if pc.upper() in ('N', 'B') else 2 if pc.upper() == 'R' else 4 if pc.upper() == 'Q' else 0
        for pc in board if pc != '.'
    )
    phase = min(phase, 24)
    total = ((mg[0] - mg[1]) * phase + (eg[0] - eg[1]) * (24 - phase)) // 24 + mat[0] - mat[1]
    return total + (15 if stm == 0 else -15)


def compute_ft_activations(fen: str, ft_weight: np.ndarray, ft_bias: np.ndarray) -> np.ndarray:
    """Compute ClippedReLU FT activations for a single position."""
    parts = fen.split()
    rows = parts[0].split('/')
    board = ['.'] * 64
    for ri, row in enumerate(rows):
        fi = 0
        for c in row:
            if c.isdigit(): fi += int(c)
            else: board[ri * 8 + fi] = c; fi += 1

    wk_f, bk_f = active_features(board)
    acc_w = ft_bias.copy()
    acc_b = ft_bias.copy()
    for f_idx in wk_f:
        if f_idx < NUM_FEATURES:
            acc_w += ft_weight[f_idx]
    for f_idx in bk_f:
        if f_idx < NUM_FEATURES:
            acc_b += ft_weight[f_idx]

    return np.clip(np.concatenate([acc_w, acc_b]), 0, 255)


def generate_weights(output_path: str = "nnue.bin", num_samples: int = 100000):
    print(f"Generating {num_samples} training positions...")
    t0 = time.time()

    # Random FT weights with proper scaling
    print("Creating random FT weights...")
    rng = np.random.RandomState(42)
    ft_weight = rng.normal(0, 0.3, (NUM_FEATURES, FT_N)).astype(np.float32)
    ft_bias = rng.normal(0, 0.1, FT_N).astype(np.float32)

    # Generate positions and compute targets
    random.seed(42)
    fens = []
    for _ in range(num_samples):
        fens.append(random_fen())

    print(f"  Generated {num_samples} fens in {time.time()-t0:.1f}s")
    t1 = time.time()

    # Compute HCE targets
    targets = np.array([hce_evaluate(f) for f in fens], dtype=np.float32)
    print(f"  HCE evaluated in {time.time()-t1:.1f}s, range: [{targets.min():.0f}, {targets.max():.0f}]")

    # Compute FT activations
    t2 = time.time()
    print("Computing FT activations (numpy)...")
    X = np.zeros((num_samples, FT_N * 2), dtype=np.float32)
    for i, fen in enumerate(fens):
        if i % 10000 == 0:
            print(f"  {i}/{num_samples} ({time.time()-t2:.1f}s)")
        X[i] = compute_ft_activations(fen, ft_weight, ft_bias)
    print(f"  FT activations computed in {time.time()-t2:.1f}s")

    # Train L1/L2 using Extreme Learning Machine approach:
    # Random L1 projection -> ReLU -> least squares L2 solve
    t3 = time.time()
    print("Training L1/L2 (ELM: random L1 + least squares L2)...")

    rng_elm = np.random.RandomState(123)
    l1_w = rng_elm.normal(0, 0.5, (FT_N * 2, L1_N)).astype(np.float32)
    l1_b = rng_elm.uniform(-1, 0, L1_N).astype(np.float32)

    # Compute L1 activations
    l1_raw = X @ l1_w + l1_b  # (N, 32)
    l1_act = np.maximum(l1_raw, 0)
    l1_act = np.clip(l1_act, 0, 127)

    # Solve L2: l1_act @ l2_w + l2_b = targets
    # Add bias column: A = [l1_act, ones]
    A = np.column_stack([l1_act, np.ones(num_samples, dtype=np.float32)])
    coeffs, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    l2_w = coeffs[:L1_N].astype(np.float32)
    l2_b = coeffs[L1_N].astype(np.float32).item()

    # Evaluate
    y_pred = l1_act @ l2_w + l2_b
    mae = np.abs(y_pred - targets).mean()
    print(f"  Training MAE: {mae:.1f}cp in {time.time()-t3:.1f}s")

    # Quantize
    print("\nQuantizing...")

    def quantize_int16(x, max_val=8191):
        a = max(np.max(np.abs(x)), 1e-6)
        s = max_val / a
        return np.clip(np.round(x * s), -32768, 32767).astype(np.int16)

    def quantize_int8(x, max_val=63):
        a = max(np.max(np.abs(x)), 1e-6)
        s = max_val / a
        return np.clip(np.round(x * s), -128, 127).astype(np.int8)

    ft_weight_q = quantize_int16(ft_weight)
    ft_bias_q = quantize_int16(ft_bias)
    l1_w_q = quantize_int8(l1_w)
    l1_b_q = quantize_int8(l1_b)
    l2_w_q = quantize_int8(l2_w)
    l2_b_q = int(np.clip(np.round(l2_b), -32768, 32767))

    with open(output_path, 'wb') as f:
        f.write(b'NNUE')
        f.write(struct.pack('<i', 1))
        f.write(struct.pack('<i', FT_N))
        f.write(struct.pack('<i', L1_N))
        f.write(ft_bias_q.tobytes())
        f.write(ft_weight_q.tobytes())
        f.write(l1_w_q.tobytes())
        f.write(l1_b_q.tobytes())
        f.write(l2_w_q.tobytes())
        f.write(struct.pack('<h', l2_b_q))

    print(f"Exported {output_path} ({os.path.getsize(output_path)} bytes)")
    print(f"  FT bias: [{ft_bias_q.min()}, {ft_bias_q.max()}]")
    print(f"  FT weight: [{ft_weight_q.min()}, {ft_weight_q.max()}]")
    print(f"  L1: [{l1_w_q.min()}, {l1_w_q.max()}]")
    print(f"  L2: [{l2_w_q.min()}, {l2_w_q.max()}]  bias={l2_b_q}")

    # Quick verification
    print("\nVerification:")
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",
        "8/8/8/4k3/8/4K3/8/8 w - - 0 1",
    ]
    for fen in test_fens:
        hce = hce_evaluate(fen)
        ft_act = compute_ft_activations(fen, ft_weight, ft_bias)
        l1_raw = ft_act @ l1_w + l1_b
        l1_out = np.maximum(l1_raw, 0)
        l1_out = np.clip(l1_out, 0, 127)
        nnue_out = float(l1_out @ l2_w + l2_b)
        cp = nnue_out * 100 / 256
        print(f"  HCE={hce:>5}  NNUE={nnue_out:>7.0f}  ={cp:>5.0f}cp")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, default=DEFAULT_EXPORT)
    parser.add_argument("--samples", type=int, default=100000)
    args = parser.parse_args()
    generate_weights(args.export, args.samples)
