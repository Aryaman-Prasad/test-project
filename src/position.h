#ifndef POSITION_H
#define POSITION_H

#include "bitboard.h"
#include "nnue.h"
#include <vector>
#include <string>

// --------------------------------------------------------------------------
// Zobrist keys
// --------------------------------------------------------------------------
extern Key ZobristPiece[12][64];
extern Key ZobristSide;
extern Key ZobristCastling[16];
extern Key ZobristEP[8];

void init_zobrist();

// --------------------------------------------------------------------------
// State info stored on the undo stack
// --------------------------------------------------------------------------
struct StateInfo {
    Key    hash_key;
    Bitboard checkers;
    int    castling_rights;
    Square en_passant_sq;
    int    halfmove_clock;
    Piece  captured_piece;
    int16_t nnue_accumulator[2][NNUE::FT_N];
};

// --------------------------------------------------------------------------
// Position
// --------------------------------------------------------------------------
class Position {
public:
    Position() { clear(); }
    explicit Position(const std::string& fen) { set_fen(fen); }

    void clear();
    void set_fen(const std::string& fen);
    void set_piece(Color c, PieceType pt, Square sq);

    // --- Accessors ---
    Bitboard pieces()                  const { return occ[WHITE] | occ[BLACK]; }
    Bitboard pieces(Color c)           const { return occ[c]; }
    Bitboard pieces(Color c, PieceType pt) const { return piece_bb[c][pt]; }
    Bitboard pieces(PieceType pt)      const { return piece_bb[WHITE][pt] | piece_bb[BLACK][pt]; }
    Bitboard occupied()                const { return pieces(); }
    Square   king_sq(Color c)          const { return lsb(piece_bb[c][KING]); }
    Color    side_to_move()            const { return stm; }
    int      get_castling_rights()     const { return castling_rights; }
    Square   ep_square()               const { return ep_sq; }
    int      halfmove_clock()          const { return rule50; }
    int      fullmove_number()         const { return fullmove_number_; }
    Key      hash_key()                const { return state_stack.empty() ? compute_hash() : state_stack.back().hash_key; }
    int      non_pawn_material(Color c) const;

    Piece    piece_on(Square sq)       const { return board[sq]; }
    Key      compute_hash()            const;

    // --- Moves ---
    void make_move(Move m);
    void unmake_move(Move m);
    void make_null_move();
    void unmake_null_move();
    bool is_legal() const;   // check if side-to-move's king is safe (call after make_move)

    // --- SEE ---
    Value see(Move m) const;

    // --- Attacks ---
    Bitboard attackers_to(Square sq, Bitboard occ) const;
    bool     is_attacked(Square sq, Color by)      const;
    Bitboard checkers()  const;
    bool     in_check()  const;
    Bitboard pinned_pieces(Color c) const;
    bool     is_draw()   const;
    int      repetition_count() const;

    std::string fen() const;
    static const std::string START_FEN;

    int16_t nnue_accumulator[2][NNUE::FT_N];

private:
    void set_piece_raw(Color c, PieceType pt, Square sq);
    void remove_piece(Color c, PieceType pt, Square sq);
    void move_piece(Color c, PieceType pt, Square from, Square to);
    void update_nnue_accumulator(Move m, Piece moved_piece, const struct StateInfo& st);

    Bitboard piece_bb[2][6];
    Bitboard occ[2];
    Piece    board[64];

    Color  stm;
    int    castling_rights;
    Square ep_sq;
    int    rule50;
    int    fullmove_number_;

    std::vector<StateInfo> state_stack;
};

#endif
