"""Evaluation and search on bitboards, compiled with numba.

agent.py's search runs on python-chess, which is 74% of its runtime. This is
the replacement: the same algorithms against bitboard.py's representation, with
nothing in the hot loop that numba has to box.
"""

import time

import numpy as np
from numba import njit, objmode

import bitboard as bb

# --- zobrist ---------------------------------------------------------------
# Random keys, drawn once at import from a fixed seed so a position hashes the
# same on every run and the table is reproducible for anyone reading the code.

_rng = np.random.default_rng(0xC0FFEE)
PIECE_KEYS = _rng.integers(0, 1 << 64, size=(12, 64), dtype=np.uint64)
CASTLING_KEYS = _rng.integers(0, 1 << 64, size=16, dtype=np.uint64)
EP_FILE_KEYS = _rng.integers(0, 1 << 64, size=8, dtype=np.uint64)
SIDE_KEY = np.uint64(_rng.integers(0, 1 << 64, dtype=np.uint64))


@njit(cache=False)
def zobrist_jit(bbs, state):
    key = np.uint64(0)
    for piece in range(12):
        pieces = bbs[piece]
        while pieces:
            square = bb.lsb_index(pieces)
            pieces &= pieces - bb.ONE
            key ^= PIECE_KEYS[piece][square]
    key ^= CASTLING_KEYS[state[bb.CASTLING]]
    if state[bb.EP] >= 0:
        key ^= EP_FILE_KEYS[state[bb.EP] % 8]
    if state[bb.SIDE] == 1:
        key ^= SIDE_KEY
    return key


def zobrist(position: bb.Position) -> int:
    return int(zobrist_jit(position[0], position[1]))


def probe_make_unmake_keeps_zobrist(position: bb.Position) -> bool:
    """Play and take back every legal move, checking the key comes home each time."""
    boards, state = position[0].copy(), position[1].copy()
    moves = np.zeros(bb.MAX_MOVES, dtype=np.int64)
    undo = np.zeros(5, dtype=np.int64)
    before = int(zobrist_jit(boards, state))
    count = bb.generate_jit(boards, state, moves)
    for index in range(count):
        move = int(moves[index])
        bb.make_jit(boards, state, move, undo)
        bb.unmake_jit(boards, state, move, undo)
        if int(zobrist_jit(boards, state)) != before:
            return False
    return True


# --- evaluation ------------------------------------------------------------
# Material and piece-square tables, tapered between a middlegame and an endgame
# king table. Same weights agent.py uses, so the two can be compared directly.

PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)
PHASE_WEIGHTS = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
PHASE_MAX = 24

_PST_SOURCE = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    2: [-20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20],
    3: [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0],
    4: [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20],
    5: [-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20],
}
_KING_ENDGAME = [
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30, -30, -50,
]

# Tables are written rank 8 first; square ^ 56 flips them into a1=0 indexing.
PST_WHITE = np.zeros((6, 64), dtype=np.int64)
PST_BLACK = np.zeros((6, 64), dtype=np.int64)
for _piece, _table in _PST_SOURCE.items():
    for _square in range(64):
        PST_WHITE[_piece][_square] = _table[_square ^ 56]
        PST_BLACK[_piece][_square] = _table[_square]
KING_EG_WHITE = np.array([_KING_ENDGAME[sq ^ 56] for sq in range(64)], dtype=np.int64)
KING_EG_BLACK = np.array([_KING_ENDGAME[sq] for sq in range(64)], dtype=np.int64)


# Pawn structure masks. Built here rather than in the kernel: they never change.
FILE_MASK = np.zeros(8, dtype=np.uint64)
ADJACENT_FILES = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    FILE_MASK[_f] = np.uint64(sum(1 << (_r * 8 + _f) for _r in range(8)))
for _f in range(8):
    _mask = 0
    if _f > 0:
        _mask |= int(FILE_MASK[_f - 1])
    if _f < 7:
        _mask |= int(FILE_MASK[_f + 1])
    ADJACENT_FILES[_f] = np.uint64(_mask)

# Squares ahead of a pawn on its own and adjacent files. Empty of enemy pawns
# means the pawn is passed.
PASSED_SPAN_WHITE = np.zeros(64, dtype=np.uint64)
PASSED_SPAN_BLACK = np.zeros(64, dtype=np.uint64)
for _sq in range(64):
    _f, _r = _sq % 8, _sq // 8
    _files = int(FILE_MASK[_f]) | int(ADJACENT_FILES[_f])
    PASSED_SPAN_WHITE[_sq] = np.uint64(
        _files & sum(0xFF << (8 * _n) for _n in range(_r + 1, 8)))
    PASSED_SPAN_BLACK[_sq] = np.uint64(
        _files & sum(0xFF << (8 * _n) for _n in range(0, _r)))

PASSED_BONUS = np.array([0, 10, 20, 35, 50, 75, 100, 0], dtype=np.int64)
DOUBLED_PENALTY = 20
ISOLATED_PENALTY = 15
BISHOP_PAIR_BONUS = 30
SHIELD_MISSING_FILE = 10
SHIELD_NEAR = 15
SHIELD_FAR = 5


@njit(cache=False)
def pawn_structure(bbs):
    """Passed, doubled and isolated pawns, from White's point of view."""
    white_pawns = bbs[0]
    black_pawns = bbs[6]
    score = 0

    pawns = white_pawns
    while pawns:
        square = bb.lsb_index(pawns)
        pawns &= pawns - bb.ONE
        if PASSED_SPAN_WHITE[square] & black_pawns == bb.ZERO:
            score += PASSED_BONUS[square >> 3]
    pawns = black_pawns
    while pawns:
        square = bb.lsb_index(pawns)
        pawns &= pawns - bb.ONE
        if PASSED_SPAN_BLACK[square] & white_pawns == bb.ZERO:
            score -= PASSED_BONUS[7 - (square >> 3)]

    for file in range(8):
        white_on_file = 0
        pawns = white_pawns & FILE_MASK[file]
        while pawns:
            pawns &= pawns - bb.ONE
            white_on_file += 1
        black_on_file = 0
        pawns = black_pawns & FILE_MASK[file]
        while pawns:
            pawns &= pawns - bb.ONE
            black_on_file += 1

        if white_on_file > 1:
            score -= DOUBLED_PENALTY * (white_on_file - 1)
        if black_on_file > 1:
            score += DOUBLED_PENALTY * (black_on_file - 1)
        if white_on_file > 0 and white_pawns & ADJACENT_FILES[file] == bb.ZERO:
            score -= ISOLATED_PENALTY * white_on_file
        if black_on_file > 0 and black_pawns & ADJACENT_FILES[file] == bb.ZERO:
            score += ISOLATED_PENALTY * black_on_file

    return score


@njit(cache=False)
def king_shelter(bbs):
    """Pawns standing in front of each king, from White's point of view."""
    score = 0
    for white in range(2):
        king = bbs[5] if white == 0 else bbs[11]
        if king == bb.ZERO:
            continue
        square = bb.lsb_index(king)
        file, rank = square % 8, square // 8
        pawns = bbs[0] if white == 0 else bbs[6]
        side = 0
        for delta in (-1, 0, 1):
            near_file = file + delta
            if near_file < 0 or near_file > 7:
                continue
            if pawns & FILE_MASK[near_file] == bb.ZERO:
                side -= SHIELD_MISSING_FILE
            step = 1 if white == 0 else -1
            for distance, bonus in ((1, SHIELD_NEAR), (2, SHIELD_FAR)):
                near_rank = rank + step * distance
                if 0 <= near_rank <= 7 and (
                    (pawns >> np.uint64(near_rank * 8 + near_file)) & bb.ONE
                ):
                    side += bonus
        score += side if white == 0 else -side
    return score


@njit(cache=False)
def evaluate_jit(bbs, state):
    """Score from the side to move's point of view, in centipawns."""
    phase_material = 0
    for piece in range(6):
        white_count = 0
        pieces = bbs[piece]
        while pieces:
            pieces &= pieces - bb.ONE
            white_count += 1
        black_count = 0
        pieces = bbs[6 + piece]
        while pieces:
            pieces &= pieces - bb.ONE
            black_count += 1
        phase_material += PHASE_WEIGHTS[piece] * (white_count + black_count)
    phase = phase_material / PHASE_MAX
    if phase > 1.0:
        phase = 1.0

    score = 0
    for piece in range(6):
        pieces = bbs[piece]
        while pieces:
            square = bb.lsb_index(pieces)
            pieces &= pieces - bb.ONE
            if piece == 5:
                table = PST_WHITE[5][square] * phase + KING_EG_WHITE[square] * (1.0 - phase)
                score += PIECE_VALUES[piece] + np.int64(table)
            else:
                score += PIECE_VALUES[piece] + PST_WHITE[piece][square]
        pieces = bbs[6 + piece]
        while pieces:
            square = bb.lsb_index(pieces)
            pieces &= pieces - bb.ONE
            if piece == 5:
                table = PST_BLACK[5][square] * phase + KING_EG_BLACK[square] * (1.0 - phase)
                score -= PIECE_VALUES[piece] + np.int64(table)
            else:
                score -= PIECE_VALUES[piece] + PST_BLACK[piece][square]

    score += pawn_structure(bbs)
    score += king_shelter(bbs)

    white_bishops = 0
    pieces = bbs[2]
    while pieces:
        pieces &= pieces - bb.ONE
        white_bishops += 1
    black_bishops = 0
    pieces = bbs[8]
    while pieces:
        pieces &= pieces - bb.ONE
        black_bishops += 1
    if white_bishops >= 2:
        score += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        score -= BISHOP_PAIR_BONUS

    return score if state[bb.SIDE] == 0 else -score


def evaluate(position: bb.Position) -> int:
    return int(evaluate_jit(position[0], position[1]))


# --- search ----------------------------------------------------------------

MATE = 900_000
MATE_BOUND = MATE - 1000
MAX_PLY = 64
TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)
EXACT, LOWER, UPPER = 0, 1, 2
NODES_PER_TIME_CHECK = 2048

TT_KEY = np.zeros(TT_SIZE, dtype=np.uint64)
TT_DEPTH = np.zeros(TT_SIZE, dtype=np.int16)
TT_FLAG = np.zeros(TT_SIZE, dtype=np.int8)
TT_MOVE = np.zeros(TT_SIZE, dtype=np.int32)
TT_SCORE = np.zeros(TT_SIZE, dtype=np.int32)

KILLERS = np.zeros((MAX_PLY, 2), dtype=np.int64)
HISTORY = np.zeros((12, 64), dtype=np.int64)

# The platform hands over a bare FEN each move, so the position arrays carry no
# history and the search cannot see a repetition coming. We keep it here: every
# position get_move is asked about, plus the current search path, as a small
# open-addressed table the jitted code can probe.
REPEAT_BITS = 12
REPEAT_SIZE = 1 << REPEAT_BITS
REPEAT_MASK = np.uint64(REPEAT_SIZE - 1)
REPEAT_KEY = np.zeros(REPEAT_SIZE, dtype=np.uint64)
REPEAT_COUNT = np.zeros(REPEAT_SIZE, dtype=np.int64)
SEEN_KEY = np.zeros(REPEAT_SIZE, dtype=np.uint64)
SEEN_COUNT = np.zeros(REPEAT_SIZE, dtype=np.int64)

# A draw is worth slightly less than nothing, so the engine does not repeat its
# way out of a won position. Applied from the root player's point of view.
CONTEMPT = 30


def new_game() -> None:
    TT_KEY[:] = 0
    TT_DEPTH[:] = 0
    TT_FLAG[:] = 0
    TT_MOVE[:] = 0
    TT_SCORE[:] = 0
    KILLERS[:] = 0
    HISTORY[:] = 0
    REPEAT_KEY[:] = 0
    REPEAT_COUNT[:] = 0
    SEEN_KEY[:] = 0
    SEEN_COUNT[:] = 0


@njit(cache=False)
def repeat_slot(key):
    """Open addressing with linear probing; the table is far larger than a game."""
    slot = np.int64(key & REPEAT_MASK)
    for _ in range(REPEAT_SIZE):
        if REPEAT_KEY[slot] == key or REPEAT_COUNT[slot] == 0:
            return slot
        slot = (slot + 1) & np.int64(REPEAT_SIZE - 1)
    return slot


@njit(cache=False)
def repeat_count(repeat_key, repeat_count_table, key):
    slot = np.int64(key & REPEAT_MASK)
    for _ in range(64):
        if repeat_count_table[slot] == 0:
            return 0
        if repeat_key[slot] == key:
            return repeat_count_table[slot]
        slot = (slot + 1) & np.int64(REPEAT_SIZE - 1)
    return 0


@njit(cache=False)
def repeat_bump(repeat_key, repeat_count_table, key, delta):
    slot = np.int64(key & REPEAT_MASK)
    for _ in range(64):
        if repeat_count_table[slot] == 0 or repeat_key[slot] == key:
            repeat_key[slot] = key
            repeat_count_table[slot] += delta
            return
        slot = (slot + 1) & np.int64(REPEAT_SIZE - 1)


@njit(cache=False)
def piece_on(bbs, square, white):
    offset = 0 if white else 6
    for index in range(6):
        if (bbs[offset + index] >> np.uint64(square)) & bb.ONE:
            return index
    return -1


@njit(cache=False)
def score_moves(bbs, state, moves, count, scores, tt_move, ply, killers,
                history):
    """Cheap ordering: the table's move, then captures by MVV-LVA, then killers."""
    white = state[bb.SIDE] == 0
    for index in range(count):
        move = moves[index]
        origin = move & 63
        target = (move >> 6) & 63
        promotion = (move >> 12) & 7
        if move == tt_move:
            scores[index] = 1_000_000
            continue
        victim = piece_on(bbs, target, not white)
        if victim >= 0:
            attacker = piece_on(bbs, origin, white)
            scores[index] = 900_000 + PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker]
        elif promotion:
            scores[index] = 880_000 + PIECE_VALUES[promotion - 1]
        elif move == killers[ply][0]:
            scores[index] = 870_000
        elif move == killers[ply][1]:
            scores[index] = 860_000
        else:
            attacker = piece_on(bbs, origin, white)
            offset = 0 if white else 6
            scores[index] = history[offset + attacker][target] if attacker >= 0 else 0


@njit(cache=False)
def pick_move(moves, scores, count, start):
    """Selection sort one step: most searches cut off long before the tail."""
    best = start
    for index in range(start + 1, count):
        if scores[index] > scores[best]:
            best = index
    moves[start], moves[best] = moves[best], moves[start]
    scores[start], scores[best] = scores[best], scores[start]
    return moves[start]


@njit(cache=False)
def null_move_allowed_jit(bbs, state):
    """Passing is only informative when the side to move has a piece to lose tempo with.

    In check a pass leaves the king capturable and the score is meaningless. In a
    king-and-pawn ending zugzwang makes passing better than every legal move, so
    the null search fails high on positions that are actually lost.
    """
    white = state[bb.SIDE] == 0
    if in_check_jit(bbs, white):
        return False
    offset = 0 if white else 6
    return (bbs[offset + 1] | bbs[offset + 2] | bbs[offset + 3] | bbs[offset + 4]) != bb.ZERO


def null_move_allowed(position: bb.Position) -> bool:
    return bool(null_move_allowed_jit(position[0], position[1]))


def probe_null_move(position: bb.Position) -> bb.Position:
    """Apply a null move and hand back the result, for tests."""
    boards, state = position[0].copy(), position[1].copy()
    state[bb.EP] = -1
    state[bb.SIDE] = 1 - state[bb.SIDE]
    state[bb.HALFMOVE] += 1
    return boards, state


@njit(cache=False)
def in_check_jit(bbs, white):
    king = bbs[(0 if white else 6) + 5]
    if king == bb.ZERO:
        return False
    return bb.is_attacked_jit(bbs, bb.lsb_index(king), not white)


@njit(cache=False)
def quiescence(bbs, state, alpha, beta, ply, move_stack, score_stack, undo_stack,
               counter, killers, history):
    counter[0] += 1
    stand_pat = evaluate_jit(bbs, state)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if ply >= MAX_PLY - 1:
        return alpha

    moves = move_stack[ply]
    scores = score_stack[ply]
    undo = undo_stack[ply]
    count = bb.generate_jit(bbs, state, moves)
    white = state[bb.SIDE] == 0
    score_moves(bbs, state, moves, count, scores, 0, ply, killers, history)

    for index in range(count):
        move = pick_move(moves, scores, count, index)
        target = (move >> 6) & 63
        flag = (move >> 15) & 3
        promotion = (move >> 12) & 7
        # Captures and promotions only, or the horizon never settles.
        if piece_on(bbs, target, not white) < 0 and flag != bb.FLAG_EP and promotion == 0:
            continue
        bb.make_jit(bbs, state, move, undo)
        if in_check_jit(bbs, white):
            bb.unmake_jit(bbs, state, move, undo)
            continue
        score = -quiescence(bbs, state, -beta, -alpha, ply + 1, move_stack, score_stack,
                            undo_stack, counter, killers, history)
        bb.unmake_jit(bbs, state, move, undo)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def negamax(bbs, state, depth, alpha, beta, ply, move_stack, score_stack, undo_stack,
            counter, deadline, timed_out,
            tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
            repeat_key, repeat_count_table, root_white):
    counter[0] += 1
    if counter[0] % NODES_PER_TIME_CHECK == 0:
        now = 0.0
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= deadline:
            timed_out[0] = 1
            return 0
    if timed_out[0]:
        return 0

    key = zobrist_jit(bbs, state)
    if ply > 0:
        if state[bb.HALFMOVE] >= 100:
            return -CONTEMPT if (state[bb.SIDE] == 0) == root_white else CONTEMPT
        if repeat_count(repeat_key, repeat_count_table, key) > 0:
            # Seen before, in this game or on this line. The referee claims the
            # third occurrence, so treat the second as the draw it will become.
            return -CONTEMPT if (state[bb.SIDE] == 0) == root_white else CONTEMPT
    slot = np.int64(key & TT_MASK)
    tt_move = 0
    if tt_key[slot] == key:
        tt_move = tt_moves[slot]
        if tt_depth[slot] >= depth and ply > 0:
            score = tt_score[slot]
            if score > MATE_BOUND:
                score -= ply
            elif score < -MATE_BOUND:
                score += ply
            flag = tt_flag[slot]
            if flag == EXACT:
                return score
            if flag == LOWER and score >= beta:
                return score
            if flag == UPPER and score <= alpha:
                return score

    white = state[bb.SIDE] == 0
    checked = in_check_jit(bbs, white)
    if checked:
        depth += 1  # check extension: a forcing line should not be cut short

    if depth <= 0:
        return quiescence(bbs, state, alpha, beta, ply, move_stack, score_stack,
                          undo_stack, counter, killers, history)

    # Null-move pruning: hand the opponent a free move. If we are still above
    # beta after giving up a tempo, the real moves will not fall below it either.
    if depth >= 3 and ply > 0 and beta < MATE_BOUND and null_move_allowed_jit(bbs, state):
        saved_ep = state[bb.EP]
        state[bb.EP] = -1
        state[bb.SIDE] = 1 - state[bb.SIDE]
        null_score = -negamax(bbs, state, depth - 3, -beta, -beta + 1, ply + 1,
                              move_stack, score_stack, undo_stack, counter, deadline,
                              timed_out, tt_key, tt_depth, tt_flag, tt_moves, tt_score,
                              killers, history,
                              repeat_key, repeat_count_table, root_white)
        state[bb.SIDE] = 1 - state[bb.SIDE]
        state[bb.EP] = saved_ep
        if timed_out[0]:
            return 0
        if null_score >= beta:
            return beta

    moves = move_stack[ply]
    scores = score_stack[ply]
    undo = undo_stack[ply]
    count = bb.generate_jit(bbs, state, moves)
    score_moves(bbs, state, moves, count, scores, tt_move, ply, killers, history)

    original_alpha = alpha
    best_score = -MATE - 1
    best_move = 0
    legal_moves = 0
    repeat_bump(repeat_key, repeat_count_table, key, 1)

    for index in range(count):
        move = pick_move(moves, scores, count, index)
        bb.make_jit(bbs, state, move, undo)
        if in_check_jit(bbs, white):
            bb.unmake_jit(bbs, state, move, undo)
            continue
        legal_moves += 1

        if legal_moves == 1:
            score = -negamax(bbs, state, depth - 1, -beta, -alpha, ply + 1,
                                         move_stack, score_stack, undo_stack, counter,
                                         deadline, timed_out,
                    tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
                    repeat_key, repeat_count_table, root_white)
        else:
            reduction = 0
            quiet = piece_on(bbs, (move >> 6) & 63, not white) < 0 and (move >> 12) & 7 == 0
            if legal_moves > 4 and depth >= 3 and quiet and not checked:
                reduction = 1
            score = -negamax(bbs, state, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1,
                             move_stack, score_stack, undo_stack, counter, deadline, timed_out,
                             tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
                    repeat_key, repeat_count_table, root_white)
            if not timed_out[0] and score > alpha and (reduction > 0 or score < beta):
                score = -negamax(bbs, state, depth - 1, -beta, -alpha, ply + 1,
                                             move_stack, score_stack, undo_stack, counter,
                                             deadline, timed_out,
                    tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
                    repeat_key, repeat_count_table, root_white)
        bb.unmake_jit(bbs, state, move, undo)

        if timed_out[0]:
            repeat_bump(repeat_key, repeat_count_table, key, -1)
            return 0
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
        if alpha >= beta:
            target = (move >> 6) & 63
            if piece_on(bbs, target, not white) < 0 and (move >> 12) & 7 == 0:
                if killers[ply][0] != move:
                    killers[ply][1] = killers[ply][0]
                    killers[ply][0] = move
                attacker = piece_on(bbs, move & 63, white)
                if attacker >= 0:
                    history[(0 if white else 6) + attacker][target] += depth * depth
            break

    repeat_bump(repeat_key, repeat_count_table, key, -1)

    if legal_moves == 0:
        return -MATE + ply if checked else 0

    stored = best_score
    if stored > MATE_BOUND:
        stored += ply
    elif stored < -MATE_BOUND:
        stored -= ply
    if tt_key[slot] != key or tt_depth[slot] <= depth:
        tt_key[slot] = key
        tt_depth[slot] = depth
        tt_score[slot] = stored
        tt_moves[slot] = best_move
        if best_score <= original_alpha:
            tt_flag[slot] = UPPER
        elif best_score >= beta:
            tt_flag[slot] = LOWER
        else:
            tt_flag[slot] = EXACT

    return best_score


@njit(cache=False)
def search_root(bbs, state, max_depth, deadline, move_stack, score_stack, undo_stack,
                counter, timed_out,
                tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
                repeat_key, repeat_count_table, root_white):
    """Iterative deepening. Keeps the last completed depth's move when time runs out."""
    best_move = 0
    moves = move_stack[0]
    undo = undo_stack[0]
    white = state[bb.SIDE] == 0

    # Something legal to return even if depth 1 does not finish.
    count = bb.generate_jit(bbs, state, moves)
    for index in range(count):
        move = moves[index]
        bb.make_jit(bbs, state, move, undo)
        legal = not in_check_jit(bbs, white)
        bb.unmake_jit(bbs, state, move, undo)
        if legal:
            best_move = move
            break
    if best_move == 0:
        return 0, 0

    completed = 0
    for depth in range(1, max_depth + 1):
        timed_out[0] = 0
        alpha = -MATE - 1
        iteration_best = 0
        count = bb.generate_jit(bbs, state, moves)
        scores = score_stack[0]
        score_moves(bbs, state, moves, count, scores, best_move, 0, killers, history)
        legal_moves = 0

        for index in range(count):
            move = pick_move(moves, scores, count, index)
            bb.make_jit(bbs, state, move, undo)
            if in_check_jit(bbs, white):
                bb.unmake_jit(bbs, state, move, undo)
                continue
            legal_moves += 1
            if legal_moves == 1:
                score = -negamax(bbs, state, depth - 1, -MATE - 1, -alpha, 1, move_stack,
                                 score_stack, undo_stack, counter, deadline, timed_out,
                                 tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
                    repeat_key, repeat_count_table, root_white)
            else:
                score = -negamax(bbs, state, depth - 1, -alpha - 1, -alpha, 1, move_stack,
                                 score_stack, undo_stack, counter, deadline, timed_out,
                                 tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
                    repeat_key, repeat_count_table, root_white)
                if not timed_out[0] and score > alpha:
                    score = -negamax(bbs, state, depth - 1, -MATE - 1, -alpha, 1, move_stack,
                                     score_stack, undo_stack, counter, deadline, timed_out,
                    tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history,
                    repeat_key, repeat_count_table, root_white)
            bb.unmake_jit(bbs, state, move, undo)
            if timed_out[0]:
                break
            if score > alpha or iteration_best == 0:
                alpha = score
                iteration_best = move

        if timed_out[0]:
            break
        if iteration_best != 0:
            best_move = iteration_best
            completed = depth
        now = 0.0
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= deadline:
            break

    return best_move, completed


def _stacks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((MAX_PLY + 2, bb.MAX_MOVES), dtype=np.int64),
        np.zeros((MAX_PLY + 2, bb.MAX_MOVES), dtype=np.int64),
        np.zeros((MAX_PLY + 2, 5), dtype=np.int64),
    )


def _run(position: bb.Position, budget_ms: int) -> tuple[int, int, int]:
    boards, state = position[0].copy(), position[1].copy()
    root_white = bool(state[bb.SIDE] == 0)
    # The search path starts from what the game has actually played through.
    REPEAT_KEY[:] = SEEN_KEY
    REPEAT_COUNT[:] = SEEN_COUNT
    move_stack, score_stack, undo_stack = _stacks()
    counter = np.zeros(1, dtype=np.int64)
    timed_out = np.zeros(1, dtype=np.int64)
    deadline = time.perf_counter() + budget_ms / 1000.0
    move, depth = search_root(
        boards, state, MAX_PLY - 2, deadline, move_stack, score_stack, undo_stack,
        counter, timed_out, TT_KEY, TT_DEPTH, TT_FLAG, TT_MOVE, TT_SCORE, KILLERS, HISTORY,
        REPEAT_KEY, REPEAT_COUNT, root_white,
    )
    return int(move), int(counter[0]), int(depth)


def search(position: bb.Position, budget_ms: int) -> str:
    move, _, _ = _run(position, budget_ms)
    return bb.move_uci(move) if move else ""


def search_nodes(position: bb.Position, budget_ms: int) -> int:
    return _run(position, budget_ms)[1]


def search_verbose(position: bb.Position, budget_ms: int) -> tuple[str, int, int]:
    move, nodes, depth = _run(position, budget_ms)
    return (bb.move_uci(move) if move else ""), nodes, depth


# A zobrist key is a full uint64 and overflows int64 crossing back into Python,
# so the key never leaves the jitted layer.


@njit(cache=False)
def remember_jit(seen_key, seen_count, bbs, state):
    repeat_bump(seen_key, seen_count, zobrist_jit(bbs, state), 1)


@njit(cache=False)
def times_seen_jit(seen_key, seen_count, bbs, state):
    return repeat_count(seen_key, seen_count, zobrist_jit(bbs, state))


def remember(position: bb.Position) -> None:
    """Record a position the game has actually reached, as get_move is handed it."""
    remember_jit(SEEN_KEY, SEEN_COUNT, position[0], position[1])


def times_seen(position: bb.Position) -> int:
    return int(times_seen_jit(SEEN_KEY, SEEN_COUNT, position[0], position[1]))


def history_size() -> int:
    return int((SEEN_COUNT != 0).sum())
