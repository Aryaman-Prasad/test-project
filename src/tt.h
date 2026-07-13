#ifndef TT_H
#define TT_H

#include "types.h"
#include <cstddef>

enum Bound : uint8_t {
    BOUND_NONE  = 0,
    BOUND_EXACT = 1,
    BOUND_LOWER = 2,
    BOUND_UPPER = 3,
};

struct TTEntry {
    uint16_t key16;
    Move     move;
    int16_t  score;
    int8_t   depth;
    uint8_t  gen_bound;  // bits 0-1: bound, bits 2-7: generation
};

class TranspositionTable {
public:
    ~TranspositionTable();
    void init(size_t mb_size);
    void clear();

    TTEntry* probe(Key key) const;
    void store(Key key, Move move, Value score, Value alpha, Value beta, Depth depth);

    void new_search() { ++generation_; }
    size_t used() const;
    size_t capacity() const { return entries_; }

    static Value tt_to_value(int16_t s, int ply);
    static int16_t value_to_tt(Value s, int ply);

private:
    TTEntry* table_ = nullptr;
    size_t entries_ = 0;
    uint8_t generation_ = 0;
};

#endif
