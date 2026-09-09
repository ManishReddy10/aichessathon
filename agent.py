"""The submission entrypoint. The platform imports this file and calls get_move."""

import time
from collections.abc import Hashable

import chess
import numpy as np
from numba import njit

# Import time runs at the game start. 90s to import packages, build tables etc.

MATE = 900_000

_PIECE_VALUE: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece-square tables from White's perspective, rank 8 (a8..h8) first, rank 1 last.
_PST_RAW: dict[int, list[int]] = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}

# In an ending the king is a fighting piece: it wants the centre, not the corner.
# The table above is a middlegame table and says the opposite, so we keep both and
# interpolate between them by how much material is left.
_PST_RAW_KING_END: list[int] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

# Phase weights: 24 with everything on, 0 once only kings and pawns remain.
_PHASE_WEIGHT: dict[int, int] = {
    chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4,
}
# Same weights indexed by piece type, so the evaluation loop can total them as it
# walks the pieces it is already walking, instead of making a second pass.
_PHASE_BY_TYPE: list[int] = [_PHASE_WEIGHT.get(pt, 0) for pt in range(7)]
_PHASE_MAX = 24

# sq ^ 56 flips the rank: a1(0)->PST row 7, a8(56)->PST row 0 (White's perspective).
# Black reuses the same table unflipped so rank 1 mirrors White's rank 8.
PST_WHITE: dict[int, list[int]] = {p: [t[sq ^ 56] for sq in range(64)] for p, t in _PST_RAW.items()}
PST_BLACK: dict[int, list[int]] = {p: [t[sq] for sq in range(64)] for p, t in _PST_RAW.items()}

# Passed pawn span: squares in front of a pawn (same + adjacent files) that must be
# free of enemy pawns for the pawn to be "passed".
# _PASSED_SPAN[color_int][square] — color_int: WHITE=1, BLACK=0
_PASSED_SPAN: list[list[int]] = [[0] * 64, [0] * 64]
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    _span_files = chess.BB_FILES[_f]
    if _f > 0:
        _span_files |= chess.BB_FILES[_f - 1]
    if _f < 7:
        _span_files |= chess.BB_FILES[_f + 1]
    for _nr in range(_r + 1, 8):
        _PASSED_SPAN[1][_sq] |= _span_files & chess.BB_RANKS[_nr]  # white looks up
    for _nr in range(0, _r):
        _PASSED_SPAN[0][_sq] |= _span_files & chess.BB_RANKS[_nr]  # black looks down

# Passed pawn bonus by rank (white perspective: rank 0=rank1, rank 6=rank7)
_PASSED_BONUS = [0, 10, 20, 35, 50, 75, 100, 0]

# ---------------------------------------------------------------------------
# Numpy arrays for the JIT evaluator.  Built once at import time.
# ---------------------------------------------------------------------------

# piece_vals[piece_type] — index 0 unused (empty square)
_NP_PIECE_VALS = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int32)

# PST lookup: _NP_PST_W[piece_type, square], _NP_PST_B[piece_type, square]
_NP_PST_W = np.zeros((7, 64), dtype=np.int32)
_NP_PST_B = np.zeros((7, 64), dtype=np.int32)
for _pt in range(1, 7):
    if _pt in PST_WHITE:
        _NP_PST_W[_pt] = PST_WHITE[_pt]
        _NP_PST_B[_pt] = PST_BLACK[_pt]

# Passed-pawn span masks as int64 bitboards (same bit pattern as uint64)
def _to_i64(x: int) -> np.int64:
    return np.int64(x if x < (1 << 63) else x - (1 << 64))

_NP_PASSED_W  = np.array([_to_i64(_PASSED_SPAN[1][sq]) for sq in range(64)], dtype=np.int64)
_NP_PASSED_B  = np.array([_to_i64(_PASSED_SPAN[0][sq]) for sq in range(64)], dtype=np.int64)
_NP_PASSED_BONUS = np.array(_PASSED_BONUS, dtype=np.int32)

# Square bitmasks and file bitmasks as int64 (bitwise ops ignore the sign)
_NP_BB_SQ    = np.array([_to_i64(1 << sq) for sq in range(64)], dtype=np.int64)
_NP_BB_FILES = np.array([_to_i64(int(chess.BB_FILES[f])) for f in range(8)], dtype=np.int64)

# Endgame king table, flipped the same way as the middlegame ones
_NP_PST_KING_EG_W = np.array([_PST_RAW_KING_END[sq ^ 56] for sq in range(64)], dtype=np.int32)
_NP_PST_KING_EG_B = np.array([_PST_RAW_KING_END[sq] for sq in range(64)], dtype=np.int32)

# Pre-allocated encoding buffers (reused every call to avoid heap churn)
_enc_pieces   = np.zeros(64, dtype=np.int32)
_enc_is_white = np.zeros(64, dtype=np.bool_)


@njit(cache=False)
def _jit_eval(
    pieces:       np.ndarray,   # int32[64]  piece type per square (0=empty)
    is_white:     np.ndarray,   # bool[64]   True if the piece is White's
    turn_white:   np.bool_,     # True if White to move
    w_pawns:      np.int64,     # bitboard
    b_pawns:      np.int64,
    w_king:       np.int32,     # square index, -1 if absent
    b_king:       np.int32,
    w_bishops:    np.int32,
    b_bishops:    np.int32,
    piece_vals:   np.ndarray,   # int32[7]
    pst_w:        np.ndarray,   # int32[7, 64]
    pst_b:        np.ndarray,
    passed_w:     np.ndarray,   # int64[64]
    passed_b:     np.ndarray,
    passed_bonus: np.ndarray,   # int32[8]
    bb_sq:        np.ndarray,   # int64[64]
    bb_files:     np.ndarray,   # int64[8]
    pst_king_eg_w: np.ndarray,  # int32[64]
    pst_king_eg_b: np.ndarray,
    phase:         np.float64,  # 1.0 = all the pieces on, 0.0 = kings and pawns
) -> np.int32:
    score      = np.int32(0)
    pawn_score = np.int32(0)
    w_file     = np.zeros(8, dtype=np.int32)
    b_file     = np.zeros(8, dtype=np.int32)

    # --- Material + PST, and per-file pawn counts + passed-pawn bonus ---
    for sq in range(64):
        pt = pieces[sq]
        if pt == np.int32(0):
            continue
        v   = piece_vals[pt]
        wh  = is_white[sq]
        if pt == np.int32(6):      # king: taper between the two tables
            mg = pst_w[pt, sq] if wh else pst_b[pt, sq]
            eg = pst_king_eg_w[sq] if wh else pst_king_eg_b[sq]
            pst = np.int32(mg * phase + eg * (1.0 - phase))
        else:
            pst = pst_w[pt, sq] if wh else pst_b[pt, sq]
        val = np.int32(v + pst)
        if wh == turn_white:
            score += val
        else:
            score -= val

        if pt == np.int32(1):          # pawn
            f = sq % 8
            if wh:
                w_file[f] += np.int32(1)
                if (passed_w[sq] & b_pawns) == np.int64(0):
                    pawn_score += passed_bonus[sq >> 3]
            else:
                b_file[f] += np.int32(1)
                if (passed_b[sq] & w_pawns) == np.int64(0):
                    pawn_score -= passed_bonus[np.int32(7) - (sq >> 3)]

    # --- Doubled and isolated pawns ---
    for f in range(8):
        wc = w_file[f]
        bc = b_file[f]
        if wc > np.int32(1):
            pawn_score -= np.int32(20) * (wc - np.int32(1))
        if bc > np.int32(1):
            pawn_score += np.int32(20) * (bc - np.int32(1))
        adj_w = np.int64(0)
        adj_b = np.int64(0)
        if f > 0:
            adj_w |= w_pawns & bb_files[f - 1]
            adj_b |= b_pawns & bb_files[f - 1]
        if f < 7:
            adj_w |= w_pawns & bb_files[f + 1]
            adj_b |= b_pawns & bb_files[f + 1]
        if wc > np.int32(0) and adj_w == np.int64(0):
            pawn_score -= np.int32(15) * wc
        if bc > np.int32(0) and adj_b == np.int64(0):
            pawn_score += np.int32(15) * bc

    score += pawn_score if turn_white else -pawn_score

    # --- King safety (pawn shield + open files) ---
    ks = np.int32(0)

    if w_king >= np.int32(0):
        wkf = w_king % np.int32(8)
        wkr = w_king // np.int32(8)
        for df in range(-1, 2):
            nf = wkf + np.int32(df)
            if nf < np.int32(0) or nf > np.int32(7):
                continue
            if (w_pawns & bb_files[nf]) == np.int64(0):
                ks -= np.int32(10)
            nr1 = wkr + np.int32(1)
            if nr1 <= np.int32(7) and (w_pawns & bb_sq[nr1 * np.int32(8) + nf]) != np.int64(0):
                ks += np.int32(15)
            nr2 = wkr + np.int32(2)
            if nr2 <= np.int32(7) and (w_pawns & bb_sq[nr2 * np.int32(8) + nf]) != np.int64(0):
                ks += np.int32(5)

    if b_king >= np.int32(0):
        bkf = b_king % np.int32(8)
        bkr = b_king // np.int32(8)
        for df in range(-1, 2):
            nf = bkf + np.int32(df)
            if nf < np.int32(0) or nf > np.int32(7):
                continue
            if (b_pawns & bb_files[nf]) == np.int64(0):
                ks += np.int32(10)
            nr1 = bkr - np.int32(1)
            if nr1 >= np.int32(0) and (b_pawns & bb_sq[nr1 * np.int32(8) + nf]) != np.int64(0):
                ks -= np.int32(15)
            nr2 = bkr - np.int32(2)
            if nr2 >= np.int32(0) and (b_pawns & bb_sq[nr2 * np.int32(8) + nf]) != np.int64(0):
                ks -= np.int32(5)

    score += ks if turn_white else -ks

    # --- Bishop pair ---
    bp = (np.int32(30) if w_bishops >= np.int32(2) else np.int32(0)) - \
         (np.int32(30) if b_bishops >= np.int32(2) else np.int32(0))
    score += bp if turn_white else -bp

    return score


EXACT, LOWER, UPPER = 0, 1, 2

# Board key for the transposition table. python-chess recomputes the polyglot
# zobrist hash from scratch on every call (11.7us); _transposition_key is the
# tuple the library already keeps for repetition detection and costs 0.56us.
# It covers pieces, side to move, castling rights and a legal en passant file.
BoardKey = Hashable

def _board_key(board: chess.Board) -> BoardKey:
    return board._transposition_key()


# Transposition table: board key -> (depth, score, flag, best_move)
TT: dict[BoardKey, tuple[int, int, int, chess.Move | None]] = {}
TT_MAX = 1_000_000

# Mate scores are stored relative to the node that found them, so a score taken
# from the table at a different ply still means "mate in N from here".
MATE_BOUND = MATE - 1000

_deadline: float = 0.0
_timeout: bool = False
_nodes: int = 0
_NODES_PER_CHECK = 2048

# The platform hands over a bare FEN each move, so chess.Board(fen) arrives with an
# empty move stack and board.is_repetition() cannot see the game's own history. We
# keep it here instead: every position get_move is asked about, counted.
_seen: dict[BoardKey, int] = {}
# _seen plus the positions on the current search path, so one lookup answers
# "have I been here before" for both the real game and the tree.
_repeat: dict[BoardKey, int] = {}
_root_turn: chess.Color = chess.WHITE

# A draw is worth slightly less than nothing, so the engine does not repeat its way
# out of a position it is winning. Scored from the root player's point of view.
CONTEMPT = 30

MAX_PLY = 64
# Two quiet moves per ply that last caused a beta cutoff there.
KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
# (from_square, to_square) -> how often that move has cut off, weighted by depth.
HISTORY: dict[tuple[int, int], int] = {}


def new_game() -> None:
    """Drop everything learned about the previous game."""
    TT.clear()
    HISTORY.clear()
    _seen.clear()
    _repeat.clear()
    for slot in KILLERS:
        slot[0] = slot[1] = None


def _phase_from_material(material: int) -> float:
    return min(material, _PHASE_MAX) / _PHASE_MAX


def _game_phase(board: chess.Board) -> float:
    """1.0 with every piece on the board, falling to 0.0 at kings and pawns."""
    material = 0
    for piece_type, weight in _PHASE_WEIGHT.items():
        material += weight * (
            board.pieces_mask(piece_type, chess.WHITE).bit_count()
            + board.pieces_mask(piece_type, chess.BLACK).bit_count()
        )
    return _phase_from_material(material)


def _evaluate(board: chess.Board) -> int:
    """Encode board in one piece_map pass, then dispatch to the JIT evaluator."""
    _enc_pieces[:] = 0
    _enc_is_white[:] = False
    w_pawns_i = 0
    b_pawns_i = 0
    w_king_sq = -1
    b_king_sq = -1
    w_bishops = 0
    b_bishops = 0
    phase_material = 0

    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        wh = piece.color == chess.WHITE
        _enc_pieces[sq]   = pt
        _enc_is_white[sq] = wh
        phase_material += _PHASE_BY_TYPE[pt]
        if pt == chess.PAWN:
            if wh:
                w_pawns_i |= 1 << sq
            else:
                b_pawns_i |= 1 << sq
        elif pt == chess.KING:
            if wh:
                w_king_sq = sq
            else:
                b_king_sq = sq
        elif pt == chess.BISHOP:
            if wh:
                w_bishops += 1
            else:
                b_bishops += 1

    return int(_jit_eval(
        _enc_pieces, _enc_is_white, np.bool_(board.turn == chess.WHITE),
        _to_i64(w_pawns_i), _to_i64(b_pawns_i),
        np.int32(w_king_sq), np.int32(b_king_sq),
        np.int32(w_bishops), np.int32(b_bishops),
        _NP_PIECE_VALS, _NP_PST_W, _NP_PST_B,
        _NP_PASSED_W, _NP_PASSED_B, _NP_PASSED_BONUS,
        _NP_BB_SQ, _NP_BB_FILES,
        _NP_PST_KING_EG_W, _NP_PST_KING_EG_B,
        np.float64(_phase_from_material(phase_material)),
    ))


def _capture_score(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: prefer capturing high-value pieces with low-value attackers."""
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    victim = board.piece_type_at(move.to_square)
    if victim is None:  # en passant
        return 10 * 100 - _PIECE_VALUE[attacker]
    return 10 * _PIECE_VALUE[victim] - _PIECE_VALUE[attacker]


def _record_cutoff(board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
    """Remember a quiet move that caused a beta cutoff, so it is tried earlier next time."""
    if board.is_capture(move) or move.promotion:
        return  # captures already sort ahead of killers on MVV-LVA
    if ply < MAX_PLY and KILLERS[ply][0] != move:
        KILLERS[ply][1] = KILLERS[ply][0]
        KILLERS[ply][0] = move
    square_pair = (move.from_square, move.to_square)
    HISTORY[square_pair] = HISTORY.get(square_pair, 0) + depth * depth


def _order_moves(
    board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
) -> None:
    first_killer, second_killer = KILLERS[ply] if ply < MAX_PLY else (None, None)

    def key(move: chess.Move) -> int:
        if move == tt_move:
            return 1_000_000
        if board.is_capture(move):
            return 900_000 + _capture_score(board, move)
        if move.promotion:
            return 880_000 + _PIECE_VALUE.get(move.promotion, 0)
        if move == first_killer:
            return 870_000
        if move == second_killer:
            return 860_000
        return HISTORY.get((move.from_square, move.to_square), 0)

    moves.sort(key=key, reverse=True)


def _draw_score(board: chess.Board) -> int:
    """What a repetition or fifty-move draw is worth, from the mover's point of view.

    Negamax scores are relative to the side to move, so the sign flips on the
    opponent's nodes. Without that flip contempt would also talk the engine out of
    a draw that saves a lost position.
    """
    return -CONTEMPT if board.turn == _root_turn else CONTEMPT


def _null_move_allowed(board: chess.Board) -> bool:
    """Passing is only informative when the side to move has a piece that can lose tempo.

    In check a null move leaves the king capturable. In a king-and-pawn ending
    zugzwang makes passing better than every legal move, so the null search fails
    high on positions that are actually lost.
    """
    if board.is_check():
        return False
    mover = board.turn
    return any(
        board.pieces_mask(piece_type, mover)
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _store(key: BoardKey, depth: int, score: int, flag: int,
           move: chess.Move | None, ply: int) -> None:
    existing = TT.get(key)
    if existing is not None and depth < existing[0] and flag != EXACT:
        return  # depth-preferred: keep the deeper entry unless this one is exact
    if existing is None and len(TT) >= TT_MAX:
        TT.clear()  # generation reset: evict everything rather than silently dropping
    TT[key] = (depth, _score_to_tt(score, ply), flag, move)


def _score_to_tt(score: int, ply: int) -> int:
    if score > MATE_BOUND:
        return score + ply
    if score < -MATE_BOUND:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score > MATE_BOUND:
        return score - ply
    if score < -MATE_BOUND:
        return score + ply
    return score


def _out_of_time() -> bool:
    global _timeout
    if time.monotonic() >= _deadline:
        _timeout = True
        return True
    return False


def _quiescence(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
    """Search captures only, so the evaluation never lands mid-exchange."""
    global _nodes
    _nodes += 1
    if _nodes % _NODES_PER_CHECK == 0 and _out_of_time():
        return 0

    stand_pat = _evaluate(board)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)

    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    captures.sort(key=lambda m: _capture_score(board, m), reverse=True)
    for move in captures:
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, ply + 1)
        board.pop()
        if _timeout:
            return 0
        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


def _alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    global _nodes
    _nodes += 1
    if _nodes % _NODES_PER_CHECK == 0 and _out_of_time():
        return 0

    key = _board_key(board)
    if ply > 0 and (_repeat.get(key) or board.halfmove_clock >= 100):
        # Seen before, in this game or on this line: treat the second visit as a draw
        # rather than waiting for the third, which is what the referee will claim.
        return _draw_score(board)
    tt_move: chess.Move | None = None
    cached = TT.get(key)
    if cached is not None:
        tt_depth, tt_score, tt_flag, tt_move = cached
        if tt_depth >= depth and ply > 0:
            score = _score_from_tt(tt_score, ply)
            if tt_flag == EXACT:
                return score
            if tt_flag == LOWER and score >= beta:
                return score
            if tt_flag == UPPER and score <= alpha:
                return score

    moves = list(board.legal_moves)
    if not moves:
        return (-MATE + ply) if board.is_check() else 0

    if depth <= 0:
        return _quiescence(board, alpha, beta, ply)

    # Null-move pruning: hand the opponent a free move. If we are still above beta
    # after giving up a tempo, the real moves will not fall below it either.
    if depth >= 3 and ply > 0 and beta < MATE_BOUND and _null_move_allowed(board):
        board.push(chess.Move.null())
        null_score = -_alpha_beta(board, depth - 3, -beta, -beta + 1, ply + 1)
        board.pop()
        if _timeout:
            return 0
        if null_score >= beta:
            return beta

    _order_moves(board, moves, tt_move, ply)

    orig_alpha = alpha
    best_score = -MATE - 1
    best_move: chess.Move | None = None
    in_check = board.is_check()
    _repeat[key] = _repeat.get(key, 0) + 1

    for index, move in enumerate(moves):
        quiet = not board.is_capture(move) and not move.promotion
        board.push(move)
        gives_check = board.is_check()

        # Late move reductions: quiet moves this far down the order are rarely best,
        # so search them shallower first and only pay full depth if one beats alpha.
        reduction = 0
        if index >= 4 and depth >= 3 and quiet and not in_check and not gives_check:
            reduction = 1

        if index == 0:
            score = -_alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
        else:
            # Principal variation search: prove the rest are worse with a null window.
            score = -_alpha_beta(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
            if not _timeout and score > alpha and (reduction or score < beta):
                score = -_alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()

        if _timeout:
            return 0
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
        if alpha >= beta:
            _record_cutoff(board, move, depth, ply)
            break

    _repeat[key] -= 1

    if not _timeout:
        flag = UPPER if best_score <= orig_alpha else (LOWER if best_score >= beta else EXACT)
        _store(key, depth, best_score, flag, best_move, ply)

    return best_score


def _search_root(board: chess.Board, depth: int, prev_best: chess.Move | None) -> chess.Move | None:
    cached = TT.get(_board_key(board))
    tt_move = (cached[3] if cached else None) or prev_best

    moves = list(board.legal_moves)
    if not moves:
        return None
    _order_moves(board, moves, tt_move, 0)

    alpha = -MATE - 1
    best_move: chess.Move | None = None

    for index, move in enumerate(moves):
        board.push(move)
        if index == 0:
            score = -_alpha_beta(board, depth - 1, -MATE - 1, -alpha, 1)
        else:
            score = -_alpha_beta(board, depth - 1, -alpha - 1, -alpha, 1)
            if not _timeout and score > alpha:
                score = -_alpha_beta(board, depth - 1, -MATE - 1, -alpha, 1)
        board.pop()
        if _timeout:
            return None
        if score > alpha or best_move is None:
            alpha = score
            best_move = move

    return best_move


def _budget_ms(time_left_ms: int) -> int:
    """Allocate from the clock we were handed. A flag loses the whole game."""
    if time_left_ms < 2000:
        return 80
    if time_left_ms < 10000:
        return time_left_ms // 8
    return min(time_left_ms // 20 + 400, 8000)


def get_move(fen: str, time_left_ms: int) -> str:
    global _deadline, _timeout, _nodes, _root_turn

    board = chess.Board(fen)
    key = _board_key(board)
    _seen[key] = _seen.get(key, 0) + 1
    _root_turn = board.turn

    moves = list(board.legal_moves)
    if not moves:
        return ""

    # The search path starts from what the game has actually played through.
    _repeat.clear()
    _repeat.update(_seen)

    _deadline = time.monotonic() + _budget_ms(time_left_ms) / 1000.0
    _timeout = False
    _nodes = 0

    best_move: chess.Move | None = None

    for depth in range(1, MAX_PLY):
        _timeout = False
        candidate = _search_root(board, depth, best_move)
        if not _timeout and candidate is not None:
            best_move = candidate
            print(f"depth {depth} nodes {_nodes} move {best_move}")
        if _timeout:
            break

    return (best_move or moves[0]).uci()


# ---------------------------------------------------------------------------
# JIT warmup — runs inside the 90s init budget before the clock starts.
# Numba compiles per unique argument-type signature; warm every type we'll
# actually pass so the first real call is instant.
# ---------------------------------------------------------------------------
_enc_pieces[0] = chess.KING
_enc_is_white[0] = True
_enc_pieces[63] = chess.KING
_enc_is_white[63] = False
_jit_eval(
    _enc_pieces, _enc_is_white, np.bool_(True),
    np.int64(0), np.int64(0), np.int32(0), np.int32(63),
    np.int32(0), np.int32(0),
    _NP_PIECE_VALS, _NP_PST_W, _NP_PST_B,
    _NP_PASSED_W, _NP_PASSED_B, _NP_PASSED_BONUS,
    _NP_BB_SQ, _NP_BB_FILES,
    _NP_PST_KING_EG_W, _NP_PST_KING_EG_B, np.float64(1.0),
)
_enc_pieces[:]   = 0
_enc_is_white[:] = False
