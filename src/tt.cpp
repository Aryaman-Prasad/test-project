#include "tt.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>

TranspositionTable::~TranspositionTable() {
    if (table_)
        std::free(table_);
}

void TranspositionTable::init(size_t mb_size) {
    if (table_)
        std::free(table_);

    size_t bytes = mb_size * 1024ULL * 1024ULL;
    entries_ = bytes / sizeof(TTEntry);
    entries_ = 1ULL << (63 - __builtin_clzll(entries_));
    if (entries_ < 1024) entries_ = 1024;

    size_t alloc_size = entries_ * sizeof(TTEntry);
    table_ = (TTEntry*)std::malloc(alloc_size);
    if (!table_) {
        entries_ = 0;
        return;
    }
    clear();
}

void TranspositionTable::clear() {
    if (table_)
        std::memset(table_, 0, entries_ * sizeof(TTEntry));
    generation_ = 0;
}

TTEntry* TranspositionTable::probe(Key key) const {
    if (!table_) return nullptr;
    uint64_t idx = key & (entries_ - 1);
    TTEntry* entry = &table_[idx];
    if ((entry->key16 == uint16_t(key >> 48)) && entry->gen_bound != 0)
        return entry;
    return nullptr;
}

void TranspositionTable::store(Key key, Move move, Value score, Value alpha, Value beta, Depth depth) {
    if (!table_) return;

    uint64_t idx = key & (entries_ - 1);
    TTEntry* entry = &table_[idx];

    uint8_t bound = BOUND_EXACT;
    if (score <= alpha)      bound = BOUND_UPPER;
    else if (score >= beta)  bound = BOUND_LOWER;

    uint16_t key16 = uint16_t(key >> 48);
    uint8_t gen_bound = (generation_ << 2) | bound;

    if (entry->key16 == key16 && entry->gen_bound != 0) {
        if (depth < entry->depth)
            return;
        entry->depth = int8_t(depth);
        entry->move  = move;
        entry->score = int16_t(score);
        entry->gen_bound = gen_bound;
        return;
    }

    uint8_t old_gen = entry->gen_bound >> 2;
    uint8_t old_bound = entry->gen_bound & 3;
    if (old_bound && old_gen == generation_ && depth <= entry->depth - 4)
        return;

    entry->key16 = key16;
    entry->move  = move;
    entry->score = int16_t(score);
    entry->depth = int8_t(depth);
    entry->gen_bound = gen_bound;
}

Value TranspositionTable::tt_to_value(int16_t s, int ply) {
    if (s > VALUE_MATE_IN_MAX_PLY)  return Value(s) - ply;
    if (s < -VALUE_MATE_IN_MAX_PLY) return Value(s) + ply;
    return s;
}

int16_t TranspositionTable::value_to_tt(Value s, int ply) {
    if (s > VALUE_MATE_IN_MAX_PLY)  return int16_t(s + ply);
    if (s < -VALUE_MATE_IN_MAX_PLY) return int16_t(s - ply);
    return int16_t(s);
}

size_t TranspositionTable::used() const {
    if (!table_) return 0;
    size_t cnt = 0;
    for (size_t i = 0; i < entries_; ++i)
        if (table_[i].gen_bound != 0)
            ++cnt;
    return cnt;
}
