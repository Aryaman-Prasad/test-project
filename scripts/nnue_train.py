"""
NNUE training pipeline for chess engine.
Architecture: HalfKP(49152) -> 256 -> 32 -> 1 (quantized to int16/int8)

Training: supervised on HCE evaluations of random legal positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import random
from typing import List, Tuple, Optional
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_EXPORT = os.path.join(PROJECT_ROOT, "nnue.bin")
DEFAULT_PT = DEFAULT_EXPORT.replace('.bin', '.pt')

# Constants matching C++ engine
FT_N = 256
L1_N = 32
NUM_FEATURES = 64 * 64 * 12  # 49152
INPUT_ACTIVATION = 255
HIDDEN_ACTIVATION = 127

PIECE_MAP = {
    'P': (0, 0, 100), 'N': (0, 1, 320), 'B': (0, 2, 330),
    'R': (0, 3, 500), 'Q': (0, 4, 900), 'K': (0, 5, 0),
    'p': (1, 0, 100), 'n': (1, 1, 320), 'b': (1, 2, 330),
    'r': (1, 3, 500), 'q': (1, 4, 900), 'k': (1, 5, 0),
}

# Piece-square tables (matching eval.cpp)
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_PST = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_PST = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
QUEEN_PST = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
KING_MID_PST = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]
KING_END_PST = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]

PST = {
    'P': PAWN_PST, 'N': KNIGHT_PST, 'B': BISHOP_PST,
    'R': ROOK_PST, 'Q': QUEEN_PST, 'K': KING_MID_PST,
}

FILES = 'abcdefgh'

def file_of(sq: int) -> int:
    return sq & 7

def rank_of(sq: int) -> int:
    return sq >> 3

def square_distance(a: int, b: int) -> int:
    fd = abs(file_of(a) - file_of(b))
    rd = abs(rank_of(a) - rank_of(b))
    return fd if fd > rd else rd


def relative_sq(sq: int, color: int) -> int:
    if color == 0:  # white
        return sq
    else:           # black
        return sq ^ 56  # rank flip


def square_from_str(s: str) -> int:
    f = ord(s[0]) - ord('a')
    r = ord(s[1]) - ord('1')
    return r * 8 + f


def square_to_str(sq: int) -> str:
    return f"{FILES[sq & 7]}{1 + (sq >> 3)}"


def fen_to_board(fen: str):
    """Parse FEN into a board[64] array (char, '.' for empty)."""
    parts = fen.split()
    board_part = parts[0]
    rows = board_part.split('/')
    board = ['.'] * 64
    for r_idx, row in enumerate(rows):
        file_idx = 0
        for c in row:
            if c.isdigit():
                file_idx += int(c)
            else:
                board[r_idx * 8 + file_idx] = c
                file_idx += 1
    return board, parts[1] if len(parts) > 1 else 'w'


def hce_evaluate(fen: str) -> int:
    """Hand-crafted evaluation matching eval.cpp. Returns score from white's perspective."""
    board, stm_str = fen_to_board(fen)
    stm = 0 if stm_str == 'w' else 1

    mat = [0, 0]
    mg = [0, 0]
    eg = [0, 0]

    for sq, piece_char in enumerate(board):
        if piece_char == '.':
            continue
        color, pt_idx, value = PIECE_MAP[piece_char]
        rsq = relative_sq(sq, color)

        mat[color] += value
        if piece_char in PST:
            mg[color] += PST[piece_char][rsq]

    # King PST (midgame + endgame)
    for color in [0, 1]:
        k_idx = 5  # king piece type
        for sq, piece_char in enumerate(board):
            if piece_char in ('K', 'k'):
                pc_color = 0 if piece_char == 'K' else 1
                if pc_color == color:
                    rsq = relative_sq(sq, color)
                    mg[color] += KING_MID_PST[rsq]
                    eg[color] += KING_END_PST[rsq]

    # Pawn structure
    for color in [0, 1]:
        pawns = [sq for sq, pc in enumerate(board) if pc == ('P' if color == 0 else 'p')]
        for sq in pawns:
            f = sq & 7
            # Doubled
            same_file = [psq for psq in pawns if (psq & 7) == f]
            if len(same_file) > 1:
                mg[color] -= 15
            # Isolated
            adj = (f > 0) + (f < 7)
            has_adj = False
            for psq in pawns:
                pf = psq & 7
                if pf == f - 1 or pf == f + 1:
                    pr = psq >> 3
                    r = sq >> 3
                    if color == 0:
                        if pr >= r:
                            has_adj = True
                    else:
                        if pr <= r:
                            has_adj = True
            if not has_adj:
                mg[color] -= 10

    # Bishop pair
    for color in [0, 1]:
        bishops = [sq for sq, pc in enumerate(board) if pc == ('B' if color == 0 else 'b')]
        if len(bishops) >= 2:
            mat[color] += 30

    # Rook on open file
    for color in [0, 1]:
        rooks = [sq for sq, pc in enumerate(board) if pc == ('R' if color == 0 else 'r')]
        for sq in rooks:
            f = sq & 7
            own_pawns_on_file = [
                psq for psq in (
                    [s for s, pc in enumerate(board) if pc == ('P' if color == 0 else 'p')]
                ) if (psq & 7) == f
            ]
            if not own_pawns_on_file:
                mg[color] += 20

    # Game phase (matching eval.cpp)
    phase = 0
    for sq, pc in enumerate(board):
        if pc == 'N' or pc == 'n':
            phase += 1
        elif pc == 'B' or pc == 'b':
            phase += 1
        elif pc == 'R' or pc == 'r':
            phase += 2
        elif pc == 'Q' or pc == 'q':
            phase += 4
    phase = min(phase, 24)

    mg_phase = phase
    eg_phase = 24 - phase

    mg_total = mg[0] - mg[1]
    eg_total = eg[0] - eg[1]
    total = (mg_total * mg_phase + eg_total * eg_phase) // 24
    total += mat[0] - mat[1]

    # Tempo for side to move
    if stm == 0:
        total += 15
    else:
        total -= 15

    return total


def active_features(fen: str) -> Tuple[List[int], List[int]]:
    """Extract HalfKP feature indices for both perspectives."""
    parts = fen.split()
    board_part = parts[0]
    rows = board_part.split('/')
    board = ['.'] * 64
    for r_idx, row in enumerate(rows):
        file_idx = 0
        for c in row:
            if c.isdigit():
                file_idx += int(c)
            else:
                board[r_idx * 8 + file_idx] = c
                file_idx += 1

    wk_sq = next(i for i, c in enumerate(board) if c == 'K')
    bk_sq = next(i for i, c in enumerate(board) if c == 'k')

    wk_features = []
    bk_features = []

    for sq, piece_char in enumerate(board):
        if piece_char == '.':
            continue
        color, pt_idx, _ = PIECE_MAP[piece_char]
        pidx = color * 6 + pt_idx

        if not (color == 0 and pt_idx == 5):
            wk_features.append((wk_sq * 64 + sq) * 12 + pidx)
        if not (color == 1 and pt_idx == 5):
            bk_features.append((bk_sq * 64 + sq) * 12 + pidx)

    return wk_features, bk_features


def random_legal_fen() -> str:
    """Generate a random legal FEN (not perfectly uniform, but good enough)."""
    # Start from startpos and make random moves
    import subprocess
    import os

    # Use the chess engine to generate random positions
    engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chess")

    fens = [Position.START_FEN]
    # We'll just use a set of pre-defined positions + shuffling
    # For now, use a set of known positions
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkb1r/pp1ppppp/2n2n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 5 4",
        "r1bqk2r/pp1pppbp/2n2np1/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 6 5",
        "r2qk2r/pp1pppbp/2n2np1/2p5/4PP2/2N2N2/PPPP2PP/R1BQKB1R b KQkq f3 0 6",
        "r2qk2r/pp1p1pbp/2n1pnp1/2p5/4PP2/2N2N2/PPPP2PP/R1BQK2R w KQkq - 0 7",
        "r2q1rk1/pp1p1pbp/2n1pnp1/2p5/4PP2/2N2N2/PPPP2PP/R1BQ1RK1 w - - 0 8",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
    ]
    random.shuffle(positions)
    return positions[:100]


class Position:
    START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    @classmethod
    def random_fen(cls) -> str:
        """Generate a random semi-legal FEN by placing pieces randomly."""
        board = ['.'] * 64
        # Place kings first
        kw = random.randint(0, 63)
        kb = random.randint(0, 63)
        while square_distance(kw, kb) < 2 or file_of(kw) == file_of(kb):
            kb = random.randint(0, 63)
        board[kw] = 'K'
        board[kb] = 'k'

        # Add other pieces randomly
        for piece, count in [('P', random.randint(0, 8)), ('p', random.randint(0, 8)),
                             ('N', random.randint(0, 2)), ('n', random.randint(0, 2)),
                             ('B', random.randint(0, 2)), ('b', random.randint(0, 2)),
                             ('R', random.randint(0, 2)), ('r', random.randint(0, 2)),
                             ('Q', random.randint(0, 1)), ('q', random.randint(0, 1))]:
            for _ in range(count):
                empty = [i for i, c in enumerate(board) if c == '.']
                if not empty:
                    break
                sq = random.choice(empty)
                color, pt_idx, _ = PIECE_MAP[piece]
                # Don't place pawns on 1st/8th ranks
                if pt_idx == 0 and (sq >> 3) in (0, 7):
                    continue
                board[sq] = piece

        # Build FEN
        rows = []
        for r in range(7, -1, -1):
            row = ''
            empty_run = 0
            for f in range(8):
                c = board[r * 8 + f]
                if c == '.':
                    empty_run += 1
                else:
                    if empty_run:
                        row += str(empty_run)
                        empty_run = 0
                    row += c
            if empty_run:
                row += str(empty_run)
            rows.append(row)
        board_str = '/'.join(rows)
        stm = 'w' if random.random() < 0.5 else 'b'

        return f"{board_str} {stm} KQkq - 0 1"

    @classmethod
    def get_all_fens(cls, count: int = 1000) -> List[str]:
        result = []
        # Use known positions
        known = [
            cls.START_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            "r1bqkb1r/pp1ppppp/2n2n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 5 4",
            "r1bqk2r/pp1pppbp/2n2np1/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 6 5",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",
            "8/8/8/4k3/8/4K3/8/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "8/8/3p4/3P4/8/8/4k3/4K3 w - - 0 1",
            "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/R1BQ1RK1 w - - 0 9",
        ]
        result.extend(known)
        # Generate random positions
        seen = set(known)
        while len(result) < count:
            fen = cls.random_fen()
            if fen not in seen:
                seen.add(fen)
                result.append(fen)
        random.shuffle(result)
        return result[:count]


class NNUE(nn.Module):
    """Full NNUE architecture in floating point."""

    def __init__(self, ft_n=FT_N, l1_n=L1_N):
        super().__init__()
        self.ft_n = ft_n
        self.l1_n = l1_n

        self.ft_weight = nn.Parameter(torch.zeros(NUM_FEATURES, ft_n))
        self.ft_bias = nn.Parameter(torch.zeros(ft_n))
        nn.init.normal_(self.ft_weight, std=0.01)

        self.l1 = nn.Linear(ft_n * 2, l1_n, bias=True)
        nn.init.normal_(self.l1.weight, std=0.1)
        nn.init.zeros_(self.l1.bias)

        self.l2 = nn.Linear(l1_n, 1, bias=True)
        nn.init.normal_(self.l2.weight, std=0.5)
        nn.init.zeros_(self.l2.bias)

    def forward(self, wk_features: List[torch.Tensor], bk_features: List[torch.Tensor]):
        batch_size = len(wk_features)
        acc_w = self.ft_bias.unsqueeze(0).expand(batch_size, -1).clone()
        acc_b = self.ft_bias.unsqueeze(0).expand(batch_size, -1).clone()

        for b in range(batch_size):
            if wk_features[b].numel() > 0:
                acc_w[b] += self.ft_weight[wk_features[b]].sum(dim=0)
            if bk_features[b].numel() > 0:
                acc_b[b] += self.ft_weight[bk_features[b]].sum(dim=0)

        clipped_w = torch.clamp(acc_w, 0, INPUT_ACTIVATION)
        clipped_b = torch.clamp(acc_b, 0, INPUT_ACTIVATION)

        combined = torch.cat([clipped_w, clipped_b], dim=1)
        l1_out = self.l1(combined)
        l1_out = F.relu(l1_out)
        out = self.l2(l1_out)

        return out.squeeze(-1)


def export_weights(model: NNUE, path: str):
    """Export trained weights to binary format for the C++ engine."""
    ft_weight = model.ft_weight.detach().cpu().numpy()  # (num_features, ft_n)
    ft_bias = model.ft_bias.detach().cpu().numpy()
    l1_weight = model.l1.weight.detach().cpu().numpy()
    l1_bias = model.l1.bias.detach().cpu().numpy()
    l2_weight = model.l2.weight.detach().cpu().numpy()
    l2_bias = model.l2.bias.detach().cpu().numpy()

    # Find scale for int16 quantization of FT
    ft_abs_max = max(np.max(np.abs(ft_weight)), np.max(np.abs(ft_bias)), 1e-6)
    ft_scale = 8191.0 / ft_abs_max  # Keep more precision

    def quantize_int16(x: np.ndarray, scale: float) -> np.ndarray:
        return np.clip(np.round(x * scale), -32768, 32767).astype(np.int16)

    def quantize_int8(x: np.ndarray) -> np.ndarray:
        abs_max = max(np.max(np.abs(x)), 1e-6)
        scale = 63.0 / abs_max
        return np.clip(np.round(x * scale), -128, 127).astype(np.int8)

    ft_weight_q = quantize_int16(ft_weight, ft_scale)
    ft_bias_q = quantize_int16(ft_bias, ft_scale)

    # L1 weights: [ft_n*2, l1_n]
    l1_weight_t = l1_weight.T  # (ft_n*2, l1_n)
    l1_weight_q = quantize_int8(l1_weight_t)
    l1_bias_q = quantize_int8(l1_bias)

    # L2 weights: [l1_n]
    l2_weight_q = quantize_int8(l2_weight.flatten())
    l2_bias_q = int(np.clip(np.round(l2_bias), -32768, 32767).astype(np.int16))

    with open(path, 'wb') as f:
        f.write(b'NNUE')
        f.write(struct.pack('<i', 1))
        f.write(struct.pack('<i', FT_N))
        f.write(struct.pack('<i', L1_N))
        f.write(ft_bias_q.tobytes())
        f.write(ft_weight_q.tobytes())  # (num_features, ft_n)
        f.write(l1_weight_q.tobytes())
        f.write(l1_bias_q.tobytes())
        f.write(l2_weight_q.tobytes())
        f.write(struct.pack('<h', l2_bias_q))

    print(f"Exported weights to {path}")
    print(f"  FT scale: {ft_scale:.2f}")
    print(f"  FT weights: [{ft_weight_q.min()}, {ft_weight_q.max()}]")
    print(f"  L1 weights: [{l1_weight_q.min()}, {l1_weight_q.max()}]")
    print(f"  L2 weights: [{l2_weight_q.min()}, {l2_weight_q.max()}]")
    print(f"  L2 bias: {l2_bias_q}")


class ChessDataset(Dataset):
    def __init__(self, fens: List[str], targets: List[float]):
        self.features_wk = []
        self.features_bk = []
        for fen in fens:
            wf, bf = active_features(fen)
            self.features_wk.append(wf)
            self.features_bk.append(bf)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (torch.tensor(self.features_wk[idx], dtype=torch.long),
                torch.tensor(self.features_bk[idx], dtype=torch.long),
                self.targets[idx])


def collate_fn(batch):
    wk_list = [item[0] for item in batch]
    bk_list = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch])
    return wk_list, bk_list, targets


def train(model=None):
    print("Generating training data...")
    fens = Position.get_all_fens()

    print(f"  Evaluating {len(fens)} positions with HCE...")
    targets = []
    for fen in fens:
        score = hce_evaluate(fen)
        targets.append(score)

    print(f"  Score range: [{min(targets)}, {max(targets)}]")

    dataset = ChessDataset(fens, targets)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    if model is None:
        model = NNUE()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_fn = nn.MSELoss()

    print("Training...")
    model.train()
    epochs = 200
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for wk_feat, bk_feat, targets_b in loader:
            optimizer.zero_grad()
            pred = model(wk_feat, bk_feat)
            loss = loss_fn(pred, targets_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(num_batches, 1)
            with torch.no_grad():
                model.eval()
                all_pred = []
                all_targ = []
                for wk_feat, bk_feat, targ_b in loader:
                    pred = model(wk_feat, bk_feat)
                    all_pred.append(pred)
                    all_targ.append(targ_b)
                all_pred = torch.cat(all_pred)
                all_targ = torch.cat(all_targ)
                mae = torch.mean(torch.abs(all_pred - all_targ))
                model.train()
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.1f}, MAE={mae.item():.1f}cp, lr={scheduler.get_last_lr()[0]:.6f}")

    return model


def verify(model: NNUE):
    """Verify NNUE predictions against HCE."""
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/R1BQ1RK1 w - - 0 9",
    ]
    print("\nVerification:")
    print(f"{'FEN':<72} {'HCE':>6} {'NNUE':>6} {'Diff':>6}")
    print("-" * 92)
    for fen in test_fens:
        hce_val = hce_evaluate(fen)
        wf_list, bf_list = active_features(fen)
        wf_t = torch.tensor(wf_list, dtype=torch.long)
        bf_t = torch.tensor(bf_list, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            nnue_val = model([wf_t], [bf_t])
            nnue_val = nnue_val.item()
        diff = abs(hce_val - nnue_val)
        marker = " ***" if diff > 100 else ""
        print(f"{fen:<72} {hce_val:>6} {nnue_val:>6.0f}  {diff:>5.0f}{marker}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, default=DEFAULT_EXPORT, help="Output weight file")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--load", type=str, default=DEFAULT_PT, help="Load checkpoint .pt file to resume training")
    args = parser.parse_args()

    model = NNUE()
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location='cpu', weights_only=True))
        print(f"Loaded checkpoint from {args.load}")

    model = train(model)
    print("\nFinal verification:")
    verify(model)

    export_weights(model, args.export)
    pt_path = args.export.replace('.bin', '.pt') if args.export.endswith('.bin') else args.export + '.pt'
    torch.save(model.state_dict(), pt_path)
    print(f"\nDone! Weights saved to {args.export}, checkpoint to {pt_path}")
