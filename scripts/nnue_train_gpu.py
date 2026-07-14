"""
NNUE training on GPU using PyTorch with CUDA.
Architecture: HalfKP(49152) -> 256 -> 32 -> 1
Trains end-to-end with HCE supervision, exports to C++ binary format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import random
import time
import os
from typing import List, Tuple

FT_N = 256
L1_N = 32
NUM_FEATURES = 64 * 64 * 12
INPUT_ACTIVATION = 255.0
HIDDEN_ACTIVATION = 127.0
BATCH_SIZE = 512
NUM_WORKERS = 4
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_EXPORT = os.path.join(PROJECT_ROOT, "nnue.bin")
DEFAULT_PT = DEFAULT_EXPORT.replace('.bin', '.pt')

PIECE_MAP = {
    'P': (0, 0), 'N': (0, 1), 'B': (0, 2),
    'R': (0, 3), 'Q': (0, 4), 'K': (0, 5),
    'p': (1, 0), 'n': (1, 1), 'b': (1, 2),
    'r': (1, 3), 'q': (1, 4), 'k': (1, 5),
}


def file_of(sq: int) -> int: return sq & 7
def rank_of(sq: int) -> int: return sq >> 3

def square_distance(a: int, b: int) -> int:
    fd = abs(file_of(a) - file_of(b))
    rd = abs(rank_of(a) - rank_of(b))
    return fd if fd > rd else rd


def relative_sq(sq: int, color: int) -> int:
    return sq if color == 0 else sq ^ 56


def random_fen() -> str:
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


# --- HCE Evaluation (Python reimplementation of eval.cpp) ---
HCE_VALUES = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0}
HCE_PST = {
    'P': [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0],
    'N': [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50],
    'B': [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20],
    'R': [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0],
    'Q': [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20],
    'K_mid': [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20],
    'K_end': [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,0,0,-10,-20,-30,-30,-10,20,30,30,20,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,20,30,30,20,-10,-30,-30,-30,0,0,0,0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50],
}


def hce_evaluate(fen: str) -> float:
    parts = fen.split()
    board_part, stm_str = parts[0], parts[1]
    rows = board_part.split('/')
    board = ['.'] * 64
    for ri, row in enumerate(rows):
        fi = 0
        for c in row:
            if c.isdigit(): fi += int(c)
            else: board[ri * 8 + fi] = c; fi += 1

    stm = 0 if stm_str == 'w' else 1
    mat = [0, 0]; mg = [0, 0]; eg = [0, 0]

    for sq, pc in enumerate(board):
        if pc == '.': continue
        color = 0 if pc.isupper() else 1
        val = HCE_VALUES.get(pc.upper(), 0)
        mat[color] += val
        pst = HCE_PST.get(pc)
        if pst:
            mg[color] += pst[relative_sq(sq, color)]

    for color in [0, 1]:
        kc = 'K' if color == 0 else 'k'
        for sq, pc in enumerate(board):
            if pc == kc:
                rsq = relative_sq(sq, color)
                mg[color] += HCE_PST['K_mid'][rsq]
                eg[color] += HCE_PST['K_end'][rsq]
                break

    for color in [0, 1]:
        pc = 'P' if color == 0 else 'p'
        pawns = [sq for sq, cp in enumerate(board) if cp == pc]
        for sq in pawns:
            f = sq & 7
            same = [psq for psq in pawns if (psq & 7) == f]
            if len(same) > 1: mg[color] -= 15
            has_adj = False
            for psq in pawns:
                pf = psq & 7
                if pf in (f - 1, f + 1):
                    if (color == 0 and (psq >> 3) >= (sq >> 3)) or \
                       (color == 1 and (psq >> 3) <= (sq >> 3)):
                        has_adj = True
            if not has_adj: mg[color] -= 10

    for color in [0, 1]:
        bc = 'B' if color == 0 else 'b'
        if sum(1 for cp in board if cp == bc) >= 2: mat[color] += 30

    for color in [0, 1]:
        rc = 'R' if color == 0 else 'r'; pc = 'P' if color == 0 else 'p'
        for sq, cp in enumerate(board):
            if cp == rc:
                f = sq & 7
                if not any((psq & 7) == f for psq, cp2 in enumerate(board) if cp2 == pc):
                    mg[color] += 20

    phase = 0
    for sq, pc in enumerate(board):
        pu = pc.upper()
        if pu in ('N', 'B'): phase += 1
        elif pu == 'R': phase += 2
        elif pu == 'Q': phase += 4
    phase = min(phase, 24)

    total = ((mg[0] - mg[1]) * phase + (eg[0] - eg[1]) * (24 - phase)) // 24
    total += mat[0] - mat[1]
    return total + (15 if stm == 0 else -15)


# --- Feature extraction ---
def active_features_from_fen(fen: str) -> Tuple[List[int], List[int]]:
    parts = fen.split()
    rows = parts[0].split('/')
    board = ['.'] * 64
    for ri, row in enumerate(rows):
        fi = 0
        for c in row:
            if c.isdigit(): fi += int(c)
            else: board[ri * 8 + fi] = c; fi += 1

    wk_sq = next(i for i, c in enumerate(board) if c == 'K')
    bk_sq = next(i for i, c in enumerate(board) if c == 'k')

    wk_feat, bk_feat = [], []
    for sq, pc in enumerate(board):
        if pc == '.': continue
        color, pt = PIECE_MAP[pc]
        pidx = color * 6 + pt
        if not (color == 0 and pt == 5):
            wk_feat.append((wk_sq * 64 + sq) * 12 + pidx)
        if not (color == 1 and pt == 5):
            bk_feat.append((bk_sq * 64 + sq) * 12 + pidx)
    return wk_feat, bk_feat


# --- Dataset ---
class ChessDataset(Dataset):
    def __init__(self, fens: List[str], targets: List[float]):
        self.features_wk = []
        self.features_bk = []
        for fen in fens:
            wf, bf = active_features_from_fen(fen)
            self.features_wk.append(wf)
            self.features_bk.append(bf)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx):
        return (torch.tensor(self.features_wk[idx], dtype=torch.long),
                torch.tensor(self.features_bk[idx], dtype=torch.long),
                self.targets[idx])


def collate_fn(batch):
    wk_list = [item[0] for item in batch]
    bk_list = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch])
    return wk_list, bk_list, targets


# --- NNUE Model ---
class NNUE(nn.Module):
    def __init__(self, ft_n=FT_N, l1_n=L1_N):
        super().__init__()
        self.ft_n = ft_n
        self.l1_n = l1_n
        self.embedding = nn.EmbeddingBag(NUM_FEATURES, ft_n, mode='sum')
        self.ft_bias = nn.Parameter(torch.zeros(ft_n))
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.l1 = nn.Linear(ft_n * 2, l1_n, bias=True)
        nn.init.normal_(self.l1.weight, std=0.1)
        nn.init.zeros_(self.l1.bias)
        self.l2 = nn.Linear(l1_n, 1, bias=True)
        nn.init.normal_(self.l2.weight, std=0.5)
        nn.init.zeros_(self.l2.bias)

    def forward(self, wk_features: List[torch.Tensor], bk_features: List[torch.Tensor]):
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
        clipped_w = torch.clamp(acc_w, 0, INPUT_ACTIVATION)
        clipped_b = torch.clamp(acc_b, 0, INPUT_ACTIVATION)
        h = torch.cat([clipped_w, clipped_b], dim=1)
        h = self.l1(h)
        h = F.relu(h)
        h = torch.clamp(h, 0, HIDDEN_ACTIVATION)
        out = self.l2(h)
        return out.squeeze(-1)


def generate_training_data(num_positions: int) -> Tuple[List[str], np.ndarray]:
    print(f"Generating {num_positions} random positions...")
    fens = []
    for i in range(num_positions):
        if i % 1000 == 0:
            print(f"  {i}/{num_positions}")
        fens.append(random_fen())

    print("Evaluating with HCE...")
    targets = []
    for i, fen in enumerate(fens):
        if i % 1000 == 0:
            print(f"  {i}/{num_positions}")
        targets.append(hce_evaluate(fen))

    return fens, np.array(targets, dtype=np.float32)


# --- Weight export ---
def export_weights(model: nn.Module, path: str, device: torch.device):
    model.eval()
    ft_weight = model.embedding.weight.detach().cpu().numpy()
    ft_bias = model.ft_bias.detach().cpu().numpy()
    l1_weight = model.l1.weight.detach().cpu().numpy().T
    l1_bias = model.l1.bias.detach().cpu().numpy()
    l2_weight = model.l2.weight.detach().cpu().numpy().flatten()
    l2_bias = model.l2.bias.detach().cpu().numpy().item()

    # Per-tensor quantization
    def scale_int16(x, max_val=8191):
        a = max(np.max(np.abs(x)), 1e-6)
        s = max_val / a
        return np.clip(np.round(x * s), -32768, 32767).astype(np.int16)

    def scale_int8(x, max_val=63):
        a = max(np.max(np.abs(x)), 1e-6)
        s = max_val / a
        return np.clip(np.round(x * s), -128, 127).astype(np.int8)

    ft_weight_q = scale_int16(ft_weight)
    ft_bias_q = scale_int16(ft_bias)
    l1_w_q = scale_int8(l1_weight)
    l1_b_q = scale_int8(l1_bias)
    l2_w_q = scale_int8(l2_weight)
    l2_b_q = int(np.clip(np.round(l2_bias), -32768, 32767))

    with open(path, 'wb') as f:
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

    print(f"Exported {path}")
    print(f"  FT: {ft_weight_q.shape}, [{ft_weight_q.min()}, {ft_weight_q.max()}]")
    print(f"  L1: {l1_w_q.shape}, [{l1_w_q.min()}, {l1_w_q.max()}]")
    print(f"  L2: {l2_w_q.shape}, [{l2_w_q.min()}, {l2_w_q.max()}]")
    size = os.path.getsize(path)
    print(f"  File size: {size} bytes")


def verify(model: nn.Module, device: torch.device):
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",
        "8/8/8/4k3/8/4K3/8/8 w - - 0 1",
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
    ]
    print("\nVerification:")
    print(f"{'FEN':<72} {'HCE':>6} {'NNUE':>6} {'cp':>6}")
    print("-" * 92)
    model.eval()
    with torch.no_grad():
        for fen in test_fens:
            hce_val = hce_evaluate(fen)
            wf, bf = active_features_from_fen(fen)
            wf_t = torch.tensor(wf, dtype=torch.long, device=device)
            bf_t = torch.tensor(bf, dtype=torch.long, device=device)
            pred = model([wf_t], [bf_t]).item()
            cp = pred * 100 / 256
            print(f"{fen:<72} {hce_val:>6} {pred:>6.0f}  {cp:>5.0f}cp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, default=DEFAULT_EXPORT)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--load", type=str, default=DEFAULT_PT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate data
    fens, targets = generate_training_data(args.samples)

    # Split
    split = int(len(fens) * 0.9)
    train_fens, val_fens = fens[:split], fens[split:]
    train_targets, val_targets = targets[:split], targets[split:]

    print(f"Train: {len(train_fens)}, Val: {len(val_fens)}")
    print(f"Score range: [{targets.min():.0f}, {targets.max():.0f}]")

    # Dataset
    train_ds = ChessDataset(train_fens, train_targets.tolist())
    val_ds = ChessDataset(val_fens, val_targets.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda')
    )

    # Model
    model = NNUE().to(device)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
        print(f"Loaded checkpoint from {args.load}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    print(f"\nTraining {args.epochs} epochs (batch_size={args.batch_size})...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for wk_feat, bk_feat, targ_b in train_loader:
            targ_b = targ_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(wk_feat, bk_feat)
            loss = loss_fn(pred, targ_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_count = 0
        with torch.no_grad():
            for wk_feat, bk_feat, targ_b in val_loader:
                targ_b = targ_b.to(device, non_blocking=True)
                pred = model(wk_feat, bk_feat)
                val_loss += loss_fn(pred, targ_b).item()
                val_mae += torch.abs(pred - targ_b).sum().item()
                val_count += targ_b.size(0)

        avg_loss = total_loss / max(num_batches, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_val_mae = val_mae / max(val_count, 1)
        t = time.time() - t0

        print(f"Epoch {epoch:2d}: train_loss={avg_loss:.0f}  val_loss={avg_val_loss:.0f}  "
              f"val_MAE={avg_val_mae:.1f}cp  lr={scheduler.get_last_lr()[0]:.6f}  {t:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    verify(model, device)
    export_weights(model, args.export, device)
    pt_path = args.export.replace('.bin', '.pt') if args.export.endswith('.bin') else args.export + '.pt'
    torch.save(model.state_dict(), pt_path)
    print(f"Checkpoint saved to {pt_path}")
    print(f"\nDone! Best val loss: {best_val_loss:.0f}")


if __name__ == "__main__":
    import argparse
    main()
