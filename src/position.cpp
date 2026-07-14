#include "position.h"
#include "nnue.h"
#include <sstream>
#include <cctype>
#include <cstring>
#include <algorithm>
#include <cstring>

// --------------------------------------------------------------------------
// Zobrist keys
// --------------------------------------------------------------------------
Key ZobristPiece[12][64];
Key ZobristSide;
Key ZobristCastling[16];
Key ZobristEP[8];

static Key splitmix64(Key& s) {
    Key z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void init_zobrist() {
    Key state = 0x123456789ABCDEFULL;
    for (int p = 0; p < 12; ++p)
        for (int sq = 0; sq < 64; ++sq)
            ZobristPiece[p][sq] = splitmix64(state);
    ZobristSide = splitmix64(state);
    for (int i = 0; i < 16; ++i) ZobristCastling[i] = splitmix64(state);
    for (int f = 0; f < 8; ++f) ZobristEP[f] = splitmix64(state);
}

// --------------------------------------------------------------------------
// FEN helpers
// --------------------------------------------------------------------------
const std::string Position::START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static Piece char_to_piece(char c) {
    switch (c) {
        case 'P': return W_PAWN;   case 'N': return W_KNIGHT;
        case 'B': return W_BISHOP; case 'R': return W_ROOK;
        case 'Q': return W_QUEEN;  case 'K': return W_KING;
        case 'p': return B_PAWN;   case 'n': return B_KNIGHT;
        case 'b': return B_BISHOP; case 'r': return B_ROOK;
        case 'q': return B_QUEEN;  case 'k': return B_KING;
        default:  return PIECE_NONE;
    }
}

static char piece_to_char(Piece p) {
    static const char tbl[] = "PNBRQKpnbrqk";
    return (p < 12) ? tbl[p] : '?';
}

// --------------------------------------------------------------------------
// Internal helpers
// --------------------------------------------------------------------------
void Position::set_piece_raw(Color c, PieceType pt, Square sq) {
    Bitboard b = square_bb(sq);
    piece_bb[c][pt] |= b;
    occ[c]          |= b;
    board[sq]        = make_piece(c, pt);
}

void Position::remove_piece(Color c, PieceType pt, Square sq) {
    Bitboard b = square_bb(sq);
    piece_bb[c][pt] ^= b;
    occ[c]          ^= b;
    board[sq]        = PIECE_NONE;
}

void Position::move_piece(Color c, PieceType pt, Square from, Square to) {
    Bitboard f = square_bb(from);
    Bitboard t = square_bb(to);
    piece_bb[c][pt] ^= f | t;
    occ[c]          ^= f | t;
    board[from]      = PIECE_NONE;
    board[to]        = make_piece(c, pt);
}

// --------------------------------------------------------------------------
// Setup
// --------------------------------------------------------------------------
void Position::clear() {
    for (auto& bb : piece_bb[0]) bb = 0;
    for (auto& bb : piece_bb[1]) bb = 0;
    occ[WHITE] = occ[BLACK] = 0;
    for (auto& sq : board) sq = PIECE_NONE;
    stm = WHITE;
    castling_rights = 0;
    ep_sq = SQ_NONE;
    rule50 = 0;
    fullmove_number_ = 1;
    state_stack.clear();
    if (NNUE::loaded)
        for (int p = 0; p < 2; ++p)
            std::memcpy(nnue_accumulator[p], NNUE::ft_bias, sizeof(NNUE::ft_bias));
}

void Position::set_piece(Color c, PieceType pt, Square sq) {
    set_piece_raw(c, pt, sq);
}

void Position::set_fen(const std::string& fen) {
    clear();
    std::istringstream ss(fen);
    std::string token;

    ss >> token;
    Square sq = A8;
    for (char c : token) {
        if (c == '/') {
            sq = Square(int(sq) - 16);
        } else if (std::isdigit(c)) {
            sq = Square(int(sq) + (c - '0'));
        } else {
            Piece p = char_to_piece(c);
            if (p != PIECE_NONE) {
                set_piece_raw(color_of(p), type_of(p), sq);
                sq = Square(int(sq) + 1);
            }
        }
    }

    ss >> token;
    stm = (token == "w") ? WHITE : BLACK;

    ss >> token;
    if (token.find('K') != std::string::npos) castling_rights |= CASTLE_WK;
    if (token.find('Q') != std::string::npos) castling_rights |= CASTLE_WQ;
    if (token.find('k') != std::string::npos) castling_rights |= CASTLE_BK;
    if (token.find('q') != std::string::npos) castling_rights |= CASTLE_BQ;

    ss >> token;
    if (token != "-") {
        File f = File(token[0] - 'a');
        Rank r = Rank(token[1] - '1');
        ep_sq = make_square(f, r);
    } else {
        ep_sq = SQ_NONE;
    }

    ss >> token;
    if (!token.empty()) rule50 = std::stoi(token);

    ss >> token;
    if (!token.empty()) fullmove_number_ = std::stoi(token);

    if (NNUE::loaded)
        NNUE::compute_full_accumulator(*this, nnue_accumulator);
}

Key Position::compute_hash() const {
    Key k = 0;
    for (Color c : {WHITE, BLACK})
        for (PieceType pt = PAWN; pt <= KING; pt = PieceType(int(pt) + 1)) {
            Bitboard bb = piece_bb[c][pt];
            while (bb) k ^= ZobristPiece[make_piece(c, pt)][pop_lsb(bb)];
        }
    if (stm == BLACK) k ^= ZobristSide;
    k ^= ZobristCastling[castling_rights & 15];
    if (ep_sq != SQ_NONE) k ^= ZobristEP[file_of(ep_sq)];
    return k;
}

// --------------------------------------------------------------------------
// Attack detection
// --------------------------------------------------------------------------
Bitboard Position::attackers_to(Square sq, Bitboard occ) const {
    Bitboard atk = 0;
    atk |= pawn_attacks(WHITE, sq) & pieces(BLACK, PAWN);
    atk |= pawn_attacks(BLACK, sq) & pieces(WHITE, PAWN);
    atk |= knight_attacks(sq) & (pieces(KNIGHT));
    atk |= bishop_attacks(sq, occ) & (pieces(BISHOP) | pieces(QUEEN));
    atk |= rook_attacks(sq, occ)   & (pieces(ROOK)   | pieces(QUEEN));
    atk |= king_attacks(sq)  & (pieces(KING));
    return atk;
}

bool Position::is_attacked(Square sq, Color by) const {
    return attackers_to(sq, occupied()) & pieces(by);
}

// --------------------------------------------------------------------------
// Check / pinned pieces
// --------------------------------------------------------------------------
Bitboard Position::checkers() const {
    Color us   = stm;
    Color them = Color(us ^ 1);
    Square ksq = king_sq(us);
    Bitboard occ_all = occupied();

    Bitboard chk = 0;
    chk |= pawn_attacks(us, ksq)    & pieces(them, PAWN);
    chk |= knight_attacks(ksq)      & pieces(them, KNIGHT);
    chk |= bishop_attacks(ksq, occ_all) & (pieces(them, BISHOP) | pieces(them, QUEEN));
    chk |= rook_attacks(ksq, occ_all)   & (pieces(them, ROOK)   | pieces(them, QUEEN));
    return chk;
}

bool Position::in_check() const { return checkers() != 0; }

Bitboard Position::pinned_pieces(Color c) const {
    Bitboard pinned = 0;
    Color us = c;
    Color them = Color(us ^ 1);
    Square ksq = king_sq(us);
    Bitboard occ_all = occupied();

    Bitboard pinners = pieces(them, ROOK) | pieces(them, QUEEN);
    Bitboard potential = rook_attacks(ksq, occ_all) & pinners;
    while (potential) {
        Square sq = pop_lsb(potential);
        Bitboard between = between_bb(ksq, sq) & occ_all;
        if (between && !more_than_one(between))
            pinned |= between & pieces(us);
    }

    pinners = pieces(them, BISHOP) | pieces(them, QUEEN);
    potential = bishop_attacks(ksq, occ_all) & pinners;
    while (potential) {
        Square sq = pop_lsb(potential);
        Bitboard between = between_bb(ksq, sq) & occ_all;
        if (between && !more_than_one(between))
            pinned |= between & pieces(us);
    }

    return pinned;
}

int Position::non_pawn_material(Color c) const {
    constexpr Value mat[5] = {0, 320, 330, 500, 900};
    int v = 0;
    for (PieceType pt = KNIGHT; pt <= QUEEN; pt = PieceType(int(pt) + 1))
        v += popcount(piece_bb[c][pt]) * mat[pt];
    return v;
}

bool Position::is_legal() const {
    Color just_moved = Color(stm ^ 1);
    Square ksq = king_sq(just_moved);
    if (ksq >= 64) return false;
    return !is_attacked(ksq, stm);
}

// --------------------------------------------------------------------------
// SEE (Static Exchange Evaluation)
// --------------------------------------------------------------------------
Value Position::see(Move m) const {
    static const Value see_val[6] = {100, 320, 330, 500, 900, 0};

    Square from = move_from(m);
    Square to   = move_to(m);
    int    type = move_type(m);

    // Value of the captured piece
    Piece captured;
    if (type == EN_PASSANT)
        captured = make_piece(Color(stm ^ 1), PAWN);
    else if (type == NORMAL || type == PROMOTION)
        captured = piece_on(to);
    else
        return 0;

    if (captured == PIECE_NONE && type != PROMOTION)
        return 0;

    Value swapList[32];
    int slIndex = 1;
    swapList[0] = see_val[type_of(captured)];

    // Promotions gain extra value
    if (type == PROMOTION)
        swapList[0] += see_val[move_promo(m)] - see_val[PAWN];

    // Remove capturing piece and captured piece from the board
    Bitboard occ = occupied();
    occ ^= square_bb(from);
    if (type == EN_PASSANT) {
        Square ep_capt = Square(int(to) + (stm == WHITE ? -8 : 8));
        occ ^= square_bb(ep_capt);
    } else {
        occ ^= square_bb(to);
    }

    // The side that will recapture
    Color stm_see = Color(stm ^ 1);

    // All pieces attacking the target square (filtered to current occupancy)
    Bitboard attackers = attackers_to(to, occ) & occ;

    while (true) {
        Bitboard stm_attackers = attackers & pieces(stm_see);
        if (!stm_attackers) break;

        // Find least valuable attacker
        PieceType pt = PAWN;
        for (; pt <= KING; pt = PieceType(int(pt) + 1)) {
            Bitboard bb = stm_attackers & pieces(stm_see, pt);
            if (bb) {
                Square attacker_sq = lsb(bb);
                occ ^= square_bb(attacker_sq);

                // Update attackers with newly discovered sliding pieces
                attackers = attackers_to(to, occ) & occ;

                swapList[slIndex] = -swapList[slIndex - 1] + see_val[pt];
                slIndex++;
                break;
            }
        }
        if (pt > KING) break;
        if (pt == KING) break;

        stm_see = Color(stm_see ^ 1);
    }

    // Backward pass (minimax)
    while (--slIndex > 0) {
        if (swapList[slIndex - 1] > -swapList[slIndex])
            swapList[slIndex - 1] = -swapList[slIndex];
    }

    return swapList[0];
}

// --------------------------------------------------------------------------
// NNUE accumulator incremental update
// --------------------------------------------------------------------------
static void apply_nnue_delta(int16_t* acc, Square ks, int delta, Piece pc, Square sq) {
    if (pc == PIECE_NONE) return;
    int pc_idx = int(pc);
    int fi = (ks * 64 + sq) * 12 + pc_idx;
    for (int i = 0; i < NNUE::FT_N; ++i)
        acc[i] = NNUE::clamp_i16(int32_t(acc[i]) + delta * int32_t(NNUE::ft_weights[fi][i]));
}

void Position::update_nnue_accumulator(Move m, Piece moved_piece, const StateInfo& st) {
    if (!NNUE::loaded) return;

    Square from = move_from(m);
    Square to = move_to(m);
    int type = move_type(m);
    Color us = color_of(moved_piece);
    PieceType pt = type_of(moved_piece);

    Square wk = king_sq(WHITE);
    Square bk = king_sq(BLACK);

    bool wk_moved = (us == WHITE) && (pt == KING);
    bool bk_moved = (us == BLACK) && (pt == KING);

    // Collect removals and additions
    struct { int delta; Piece pc; Square sq; } deltas[6];
    int nd = 0;

    // Moving piece
    deltas[nd++] = {-1, moved_piece, from};
    deltas[nd++] = {+1, board[to], to};

    // Captured piece
    Square cap_sq = to;
    if (type == EN_PASSANT)
        cap_sq = Square(int(to) + (us == WHITE ? -8 : 8));
    if (st.captured_piece != PIECE_NONE)
        deltas[nd++] = {-1, st.captured_piece, cap_sq};

    // Castling rook
    if (type == CASTLING) {
        Square rf = SQ_NONE, rt = SQ_NONE;
        if (to == G1) { rf = H1; rt = F1; }
        else if (to == C1) { rf = A1; rt = D1; }
        else if (to == G8) { rf = H8; rt = F8; }
        else if (to == C8) { rf = A8; rt = D8; }
        if (rf != SQ_NONE) {
            deltas[nd++] = {-1, make_piece(us, ROOK), rf};
            deltas[nd++] = {+1, make_piece(us, ROOK), rt};
        }
    }

    // Apply to each perspective separately
    if (wk_moved)
        NNUE::compute_persp_accumulator(*this, 0, nnue_accumulator[0]);

    if (bk_moved)
        NNUE::compute_persp_accumulator(*this, 1, nnue_accumulator[1]);

    // Apply deltas to perspectives whose king did NOT move
    for (int p = 0; p < 2; ++p) {
        bool king_moved = (p == 0) ? wk_moved : bk_moved;
        if (king_moved) continue;

        Square ks = (p == 0) ? wk : bk;
        Piece my_king = (p == 0) ? W_KING : B_KING;
        for (int i = 0; i < nd; ++i) {
            if (deltas[i].pc == my_king) continue;
            apply_nnue_delta(nnue_accumulator[p], ks, deltas[i].delta, deltas[i].pc, deltas[i].sq);
        }
    }
}

// --------------------------------------------------------------------------
// make_move / unmake_move
// --------------------------------------------------------------------------
void Position::make_move(Move m) {
    Square from = move_from(m);
    Square to   = move_to(m);
    int    type = move_type(m);
    PieceType promo = move_promo(m);

    Color us   = stm;
    Color them = Color(us ^ 1);
    Piece moved_piece = board[from];
    PieceType pt = type_of(moved_piece);

    // Save state
    StateInfo st;
    st.checkers        = checkers();
    st.castling_rights = castling_rights;
    st.en_passant_sq   = ep_sq;
    st.halfmove_clock  = rule50;
    st.captured_piece  = board[to];
    if (NNUE::loaded)
        std::memcpy(st.nnue_accumulator, nnue_accumulator, sizeof(nnue_accumulator));

    // EP: capture square is behind the target
    if (type == EN_PASSANT) {
        Square ep_capt = Square(int(to) + (us == WHITE ? -8 : 8));
        st.captured_piece = board[ep_capt];
    }

    state_stack.push_back(st);

    // --- Clear EP ---
    ep_sq = SQ_NONE;

    // --- Update castling rights ---
    castling_rights &= CastleMask[from] & CastleMask[to];

    // --- Execute move on the board ---
    if (type == CASTLING) {
        move_piece(us, KING, from, to);
        if (to == G1) move_piece(us, ROOK, H1, F1);
        else if (to == C1) move_piece(us, ROOK, A1, D1);
        else if (to == G8) move_piece(us, ROOK, H8, F8);
        else if (to == C8) move_piece(us, ROOK, A8, D8);
    } else if (type == PROMOTION) {
        // Remove captured piece if promotion capture
        if (st.captured_piece != PIECE_NONE && to != from) {
            PieceType capt_pt = type_of(st.captured_piece);
            Color capt_col = color_of(st.captured_piece);
            remove_piece(capt_col, capt_pt, to);
        }
        remove_piece(us, PAWN, from);
        set_piece_raw(us, promo, to);
    } else {
        // Normal or EP
        if (type == EN_PASSANT) {
            Square ep_capt = Square(int(to) + (us == WHITE ? -8 : 8));
            remove_piece(them, PAWN, ep_capt);
        } else if (st.captured_piece != PIECE_NONE) {
            PieceType capt_pt = type_of(st.captured_piece);
            Color capt_col = color_of(st.captured_piece);
            remove_piece(capt_col, capt_pt, to);
        }
        move_piece(us, pt, from, to);
    }

    // --- Update NNUE accumulator ---
    update_nnue_accumulator(m, moved_piece, state_stack.back());

    // --- Set EP for double pawn push ---
    if (pt == PAWN && type != PROMOTION) {
        int diff = int(to) - int(from);
        if (diff == 16 || diff == -16)
            ep_sq = Square(int(from) + diff / 2);
    }

    // --- Rule50 ---
    if (pt == PAWN || st.captured_piece != PIECE_NONE)
        rule50 = 0;
    else
        ++rule50;

    // --- Switch side ---
    stm = them;
    if (stm == WHITE) ++fullmove_number_;

    // --- Compute hash after the move ---
    state_stack.back().hash_key = compute_hash();
}

void Position::unmake_move(Move m) {
    if (state_stack.empty()) return;

    StateInfo st = state_stack.back();
    state_stack.pop_back();

    Square from   = move_from(m);
    Square to     = move_to(m);
    int    type   = move_type(m);
    PieceType promo = move_promo(m);

    Color us   = Color(stm ^ 1); // side that just moved
    Color them = stm;            // side to move (before unmake)

    // Restore simple state
    castling_rights = st.castling_rights;
    ep_sq           = st.en_passant_sq;
    rule50          = st.halfmove_clock;
    if (them == WHITE) --fullmove_number_;

    // Restore NNUE accumulator
    if (NNUE::loaded)
        std::memcpy(nnue_accumulator, st.nnue_accumulator, sizeof(nnue_accumulator));

    // Undo board changes
    if (type == CASTLING) {
        move_piece(us, KING, to, from);
        if (to == G1) move_piece(us, ROOK, F1, H1);
        else if (to == C1) move_piece(us, ROOK, D1, A1);
        else if (to == G8) move_piece(us, ROOK, F8, H8);
        else if (to == C8) move_piece(us, ROOK, D8, A8);
    } else if (type == PROMOTION) {
        remove_piece(us, promo, to);
        set_piece_raw(us, PAWN, from);
        if (st.captured_piece != PIECE_NONE)
            set_piece_raw(color_of(st.captured_piece), type_of(st.captured_piece), to);
    } else {
        // Undo the piece move
        move_piece(us, type_of(board[to]), to, from);

        // Restore captured piece
        if (type == EN_PASSANT) {
            Square ep_capt = Square(int(to) + (us == WHITE ? -8 : 8));
            set_piece_raw(them, PAWN, ep_capt);
        } else if (st.captured_piece != PIECE_NONE) {
            set_piece_raw(color_of(st.captured_piece), type_of(st.captured_piece), to);
        }
    }

    stm = us;
}

void Position::make_null_move() {
    StateInfo st;
    st.hash_key        = hash_key();
    st.checkers        = checkers();
    st.castling_rights = castling_rights;
    st.en_passant_sq   = ep_sq;
    st.halfmove_clock  = rule50;
    st.captured_piece  = PIECE_NONE;
    if (NNUE::loaded)
        std::memcpy(st.nnue_accumulator, nnue_accumulator, sizeof(nnue_accumulator));
    state_stack.push_back(st);

    ep_sq = SQ_NONE;
    stm = Color(stm ^ 1);

    Key k = state_stack.back().hash_key;
    k ^= ZobristSide;
    if (st.en_passant_sq != SQ_NONE)
        k ^= ZobristEP[file_of(st.en_passant_sq)];
    state_stack.back().hash_key = k;
}

void Position::unmake_null_move() {
    if (state_stack.empty()) return;
    StateInfo st = state_stack.back();
    state_stack.pop_back();

    castling_rights = st.castling_rights;
    ep_sq           = st.en_passant_sq;
    rule50          = st.halfmove_clock;
    stm = Color(stm ^ 1);
}

// --------------------------------------------------------------------------
// Repetition / draw
// --------------------------------------------------------------------------
int Position::repetition_count() const {
    int cnt = 1;
    Key h = hash_key();
    int start = std::max(0, (int)state_stack.size() - rule50);
    for (int i = (int)state_stack.size() - 1; i >= start; --i) {
        if (state_stack[i].hash_key == h) ++cnt;
    }
    return cnt;
}

bool Position::is_draw() const {
    if (rule50 >= 100) return true;
    if (repetition_count() >= 3) return true;

    Bitboard all = pieces();
    if (all == (square_bb(king_sq(WHITE)) | square_bb(king_sq(BLACK))))
        return true;
    if (popcount(all) == 3) {
        if (pieces(KNIGHT) || pieces(BISHOP))
            return true;
    }
    return false;
}

// --------------------------------------------------------------------------
// FEN generation
// --------------------------------------------------------------------------
std::string Position::fen() const {
    std::string out;
    for (Rank r = RANK_8; r >= RANK_1; r = Rank(int(r) - 1)) {
        int empty = 0;
        for (File f = FILE_A; f <= FILE_H; f = File(int(f) + 1)) {
            Square sq = make_square(f, r);
            Piece p = board[sq];
            if (p == PIECE_NONE) { ++empty; }
            else {
                if (empty) { out += char('0' + empty); empty = 0; }
                out += piece_to_char(p);
            }
        }
        if (empty) out += char('0' + empty);
        if (r > RANK_1) out += '/';
    }
    out += (stm == WHITE) ? " w " : " b ";
    if (castling_rights == 0) out += '-';
    else {
        if (castling_rights & CASTLE_WK) out += 'K';
        if (castling_rights & CASTLE_WQ) out += 'Q';
        if (castling_rights & CASTLE_BK) out += 'k';
        if (castling_rights & CASTLE_BQ) out += 'q';
    }
    out += ' ';
    if (ep_sq == SQ_NONE) out += '-';
    else {
        out += char('a' + file_of(ep_sq));
        out += char('1' + rank_of(ep_sq));
    }
    out += ' ' + std::to_string(rule50);
    out += ' ' + std::to_string(fullmove_number_);
    return out;
}
