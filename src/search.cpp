#include "search.h"
#include "eval.h"
#include "movegen.h"
#include "tt.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <chrono>

std::atomic<bool> SearchAborted{false};
TranspositionTable TT;
static uint64_t Nodes = 0;

static const int MVV_LVA[6][6] = {
    { 0,  0,  0,  0,  0,  0},
    {23, 24, 25, 26, 27, 28},
    {17, 18, 19, 20, 21, 22},
    {11, 12, 13, 14, 15, 16},
    { 5,  6,  7,  8,  9, 10},
    { 0,  1,  2,  3,  4,  5},
};

static const Value PieceValues[6] = {100, 320, 330, 500, 900, 20000};

static Move KillerMoves[MAX_PLY][2];
static int History[2][64][64];

void init_search() {
    std::memset(KillerMoves, 0, sizeof(KillerMoves));
    std::memset(History, 0, sizeof(History));
    SearchAborted = false;
    TT.init(64);
}

static int move_order_score(Move m, const Position& pos, int ply, Move tt_move) {
    if (m == tt_move)
        return 10000;

    int type = move_type(m);
    if (type == CASTLING)
        return 5000;
    if (type == PROMOTION)
        return 4000 + PieceValues[move_promo(m)];

    Square from = move_from(m);
    Square to = move_to(m);
    Piece captured = pos.piece_on(to);

    if (captured != PIECE_NONE) {
        PieceType attacker = type_of(pos.piece_on(from));
        PieceType victim = type_of(captured);
        return 3000 + MVV_LVA[attacker][victim];
    }

    if (m == KillerMoves[ply][0] || m == KillerMoves[ply][1])
        return 2000;

    Color c = pos.side_to_move();
    return History[c][from][to];
}

static void sort_moves(MoveList& list, const Position& pos, int ply, Move tt_move) {
    int scores[256];
    for (int i = 0; i < list.count; ++i)
        scores[i] = move_order_score(list.moves[i], pos, ply, tt_move);

    for (int i = 1; i < list.count; ++i)
        for (int j = i; j > 0 && scores[j] > scores[j - 1]; --j) {
            std::swap(scores[j], scores[j - 1]);
            std::swap(list.moves[j], list.moves[j - 1]);
        }
}

static Value quiescence(Position& pos, Value alpha, Value beta, Depth depth) {
    if (SearchAborted) return 0;

    Value stand_pat = evaluate(pos);

    if (stand_pat >= beta)
        return beta;
    if (stand_pat > alpha)
        alpha = stand_pat;

    if (depth <= 0)
        return stand_pat;

    MoveList list;
    generate_moves(pos, list);

    MoveList capt_list;
    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        int type = move_type(m);
        if (type == PROMOTION || type == EN_PASSANT
            || (type == NORMAL && pos.piece_on(move_to(m)) != PIECE_NONE))
            capt_list.add(m);
    }

    if (capt_list.count == 0)
        return stand_pat;

    int scores[256];
    for (int i = 0; i < capt_list.count; ++i) {
        Move m = capt_list.moves[i];
        if (move_type(m) == PROMOTION) {
            scores[i] = 5000 + PieceValues[move_promo(m)];
        } else {
            Square from = move_from(m);
            Square to = move_to(m);
            Piece captured = pos.piece_on(to);
            if (captured != PIECE_NONE) {
                PieceType attacker = type_of(pos.piece_on(from));
                PieceType victim = type_of(captured);
                scores[i] = 3000 + MVV_LVA[attacker][victim];
            } else {
                scores[i] = 2000;
            }
        }
    }

    for (int i = 1; i < capt_list.count; ++i)
        for (int j = i; j > 0 && scores[j] > scores[j - 1]; --j) {
            std::swap(scores[j], scores[j - 1]);
            std::swap(capt_list.moves[j], capt_list.moves[j - 1]);
        }

    for (int i = 0; i < capt_list.count; ++i) {
        Move m = capt_list.moves[i];
        int mtype = move_type(m);
        // Prune losing captures (SEE < 0) at shallow depths
        if (depth <= 2 && mtype != PROMOTION && mtype != EN_PASSANT) {
            if (pos.see(m) < 0)
                continue;
        }
        pos.make_move(m);
        Nodes++;
        if (pos.is_legal()) {
            Value score = -quiescence(pos, -beta, -alpha, depth - 1);
            pos.unmake_move(m);
            if (SearchAborted) return 0;
            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        } else {
            pos.unmake_move(m);
        }
    }

    return alpha;
}

static Value alpha_beta(Position& pos, Value alpha, Value beta, Depth depth, Depth ply, Move& best_move) {
    if (depth <= 0)
        return quiescence(pos, alpha, beta, 4);

    if (SearchAborted) return 0;

    if (pos.is_draw())
        return 0;

    if (alpha < -VALUE_MATE + MAX_PLY) alpha = -VALUE_MATE + MAX_PLY;
    if (beta > VALUE_MATE - MAX_PLY)  beta = VALUE_MATE - MAX_PLY;
    if (alpha >= beta) return alpha;

    bool in_check = pos.in_check();
    if (in_check)
        depth++;

    Value static_eval = evaluate(pos);

    // TT probe
    Move tt_move = MOVE_NONE;
    Key hash_key = pos.hash_key();
    TTEntry* tt_entry = TT.probe(hash_key);
    if (tt_entry && tt_entry->depth >= depth) {
        if (tt_entry->gen_bound & 3) {
            Value tt_score = TranspositionTable::tt_to_value(tt_entry->score, ply);
            uint8_t bound = tt_entry->gen_bound & 3;
            if (bound == BOUND_EXACT)
                return tt_score;
            else if (bound == BOUND_LOWER && tt_score >= beta)
                return tt_score;
            else if (bound == BOUND_UPPER && tt_score <= alpha)
                return tt_score;
        }
    }
    if (tt_entry && tt_entry->move != MOVE_NONE)
        tt_move = tt_entry->move;

    // Null Move Pruning
    if (depth >= 3 && !in_check && beta < VALUE_MATE_IN_MAX_PLY) {
        Color us = pos.side_to_move();
        if (pos.non_pawn_material(us) > 0) {
            Depth R = 2 + depth / 6;
            if (R > 3) R = 3;
            pos.make_null_move();
            Nodes++;
            Move child_best = MOVE_NONE;
            Value null_score = -alpha_beta(pos, -beta, -beta + 1, depth - R - 1, ply + 1, child_best);
            pos.unmake_null_move();
            if (SearchAborted) return 0;
            if (null_score >= beta) {
                if (null_score < VALUE_MATE_IN_MAX_PLY)
                    return beta;
            }
        }
    }

    MoveList list;
    generate_moves(pos, list);

    if (list.count == 0) {
        if (in_check)
            return -VALUE_MATE + ply;
        return 0;
    }

    sort_moves(list, pos, ply, tt_move);

    best_move = MOVE_NONE;
    Value orig_alpha = alpha;
    int moves_searched = 0;

    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        Move child_best = MOVE_NONE;

        // --- Futility pruning for quiet moves ---
        if (moves_searched > 0 && depth <= 8 && !in_check
            && move_type(m) == NORMAL && pos.piece_on(move_to(m)) == PIECE_NONE
            && static_eval + 100 * depth < alpha) {
            moves_searched++;
            continue;
        }

        pos.make_move(m);
        Nodes++;
        if (pos.is_legal()) {
            Value score;
            if (moves_searched == 0) {
                score = -alpha_beta(pos, -beta, -alpha, depth - 1, ply + 1, child_best);
            } else {
                if (moves_searched >= 4 && depth >= 3 && !in_check
                    && move_type(m) != PROMOTION && move_type(m) != CASTLING
                    && pos.piece_on(move_to(m)) == PIECE_NONE) {
                    Depth reduction = (moves_searched >= 6) ? 2 : 1;
                    score = -alpha_beta(pos, -alpha - 1, -alpha, depth - reduction - 1, ply + 1, child_best);
                    if (score > alpha)
                        score = -alpha_beta(pos, -beta, -alpha, depth - 1, ply + 1, child_best);
                } else {
                    score = -alpha_beta(pos, -alpha - 1, -alpha, depth - 1, ply + 1, child_best);
                    if (score > alpha && score < beta)
                        score = -alpha_beta(pos, -beta, -alpha, depth - 1, ply + 1, child_best);
                }
            }
            pos.unmake_move(m);
            if (SearchAborted) return 0;

            if (score > alpha) {
                alpha = score;
                best_move = m;
                if (score >= beta) {
                    if (move_type(m) != PROMOTION && pos.piece_on(move_to(m)) == PIECE_NONE) {
                        KillerMoves[ply][1] = KillerMoves[ply][0];
                        KillerMoves[ply][0] = m;
                        Color c = pos.side_to_move();
                        History[c][move_from(m)][move_to(m)] += depth * depth;
                        if (History[c][move_from(m)][move_to(m)] > 30000)
                            for (int i = 0; i < 64; ++i)
                                for (int j = 0; j < 64; ++j)
                                    History[c][i][j] /= 2;
                    }
                    break;
                }
            }
        } else {
            pos.unmake_move(m);
        }
        moves_searched++;
    }

    // Store in TT
    int16_t stored_score = TranspositionTable::value_to_tt(alpha, ply);
    TT.store(hash_key, best_move, stored_score, orig_alpha, beta, depth);

    return alpha;
}

void search(Position& pos, SearchLimits& limits, SearchInfo& info) {
    SearchAborted = false;
    init_search();
    TT.new_search();

    auto start_time = std::chrono::steady_clock::now();
        Nodes = 0;
    info.nodes = 0;
    info.completed = false;

    Depth max_depth = limits.depth;
    if (max_depth > MAX_PLY) max_depth = MAX_PLY;

    int64_t time_limit = limits.movetime;
    if (time_limit == 0 && limits.time_left[WHITE] > 0) {
        Color stm = pos.side_to_move();
        time_limit = limits.time_left[stm] / 40;
        if (limits.inc[stm] > 0)
            time_limit += limits.inc[stm] * 3 / 4;
        if (time_limit > limits.time_left[stm] / 2)
            time_limit = limits.time_left[stm] / 2;
        if (time_limit < 10) time_limit = 10;
    }
    if (time_limit == 0) time_limit = 60000;
    int64_t deadline = time_limit;

    Move best_move = MOVE_NONE;
    Value best_score = VALUE_NONE;
    Value prev_score = VALUE_NONE;

    for (Depth d = 1; d <= max_depth; ++d) {
        Move iter_best = MOVE_NONE;
        Value score;

        if (d >= 3 && prev_score != VALUE_NONE) {
            Value delta = 15 + d * 5;
            Value alpha = prev_score - delta;
            if (alpha < -VALUE_MATE) alpha = -VALUE_MATE;
            Value beta = prev_score + delta;
            if (beta > VALUE_MATE) beta = VALUE_MATE;

            score = alpha_beta(pos, alpha, beta, d, 0, iter_best);

            if (score <= alpha)
                score = alpha_beta(pos, -VALUE_MATE, beta, d, 0, iter_best);
            else if (score >= beta)
                score = alpha_beta(pos, alpha, VALUE_MATE, d, 0, iter_best);
        } else {
            score = alpha_beta(pos, -VALUE_MATE, VALUE_MATE, d, 0, iter_best);
        }

        prev_score = score;

        if (SearchAborted)
            break;

        if (iter_best != MOVE_NONE) {
            best_move = iter_best;
            best_score = score;
        }

        auto now = std::chrono::steady_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        info.nodes = Nodes;
        info.depth = d;
        info.score = score;
        info.time_ms = elapsed;

        // Extract PV and print info
        Move pv[MAX_PLY];
        int pv_len = 0;
        {
            Position pos_copy = pos;
            Key root_key = pos_copy.hash_key();
            Key k = root_key;
            while (pv_len < MAX_PLY - 1) {
                TTEntry* e = TT.probe(k);
                if (!e) { break; }
                if (e->move == MOVE_NONE) { break; }
                MoveList ml;
                generate_moves(pos_copy, ml);
                bool found = false;
                for (int i = 0; i < ml.count; ++i) {
                    if (ml.moves[i] == e->move) { found = true; break; }
                }
                if (!found) break;
                pv[pv_len++] = e->move;
                pos_copy.make_move(e->move);
                k = pos_copy.hash_key();
            }
        }

        char pv_str[256] = {0};
        int pos_ = 0;
        for (int i = 0; i < pv_len; ++i) {
            Move m = pv[i];
            if (move_type(m) == PROMOTION)
                pos_ += snprintf(pv_str + pos_, sizeof(pv_str) - pos_, " %c%c%c%c%c",
                    'a' + file_of(move_from(m)), '1' + rank_of(move_from(m)),
                    'a' + file_of(move_to(m)), '1' + rank_of(move_to(m)),
                    "pnbrqk"[move_promo(m)]);
            else
                pos_ += snprintf(pv_str + pos_, sizeof(pv_str) - pos_, " %c%c%c%c",
                    'a' + file_of(move_from(m)), '1' + rank_of(move_from(m)),
                    'a' + file_of(move_to(m)), '1' + rank_of(move_to(m)));
            if (pos_ >= (int)sizeof(pv_str) - 20) break;
        }

        std::string score_str;
        if (score > VALUE_MATE_IN_MAX_PLY)
            score_str = "mate " + std::to_string((VALUE_MATE - score + 1) / 2);
        else if (score < -VALUE_MATE_IN_MAX_PLY)
            score_str = "mate -" + std::to_string((VALUE_MATE + score) / 2);
        else
            score_str = "cp " + std::to_string(score);

        std::cout << "info depth " << d << " score " << score_str
                  << " nodes " << info.nodes << " time " << elapsed
                  << " pv" << pv_str << std::endl;

        if (time_limit > 0 && elapsed >= deadline)
            break;

        if (score > VALUE_MATE_IN_MAX_PLY || score < -VALUE_MATE_IN_MAX_PLY)
            break;

        if (time_limit > 0 && elapsed * 3 > time_limit)
            break;
    }

    info.completed = !SearchAborted;

    if (best_move == MOVE_NONE) {
        MoveList list;
        generate_moves(pos, list);
        for (int i = 0; i < list.count; ++i) {
            Move m = list.moves[i];
            pos.make_move(m);
            if (pos.is_legal()) {
                best_move = m;
                pos.unmake_move(m);
                break;
            }
            pos.unmake_move(m);
        }
    }

    info.best_move = best_move;
}
