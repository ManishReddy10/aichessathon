"""The submission entrypoint. The platform imports this file and calls get_move.

The search itself lives in engine.py, on the bitboard representation in
bitboard.py. Both compile under numba; this file is the adapter between the
platform's contract (a FEN in, a UCI string out) and that search.
"""

import time

import bitboard as bb
import engine

# Import time runs at the game start, with a 90s budget before the clock starts.
# Everything expensive belongs here: the attack tables, and warming every jitted
# function so numba compiles now rather than on the clock.

MIN_BUDGET_MS = 80
MAX_BUDGET_MS = 8_000


def new_game() -> None:
    """Drop everything learned about the previous game."""
    engine.new_game()


def budget_ms(time_left_ms: int) -> int:
    """How long to think. A flag loses the whole game, so this stays conservative.

    The increment lands after the move, so a little of it can be spent up front.
    """
    if time_left_ms < 2_000:
        return MIN_BUDGET_MS
    if time_left_ms < 10_000:
        return time_left_ms // 8
    return min(time_left_ms // 20 + 400, MAX_BUDGET_MS)


def get_move(fen: str, time_left_ms: int) -> str:
    position = bb.from_fen(fen)

    # The platform hands over a bare FEN, so the game's own history has to be
    # kept here or the search walks into a repetition without seeing it.
    engine.remember(position)

    legal = bb.generate_legal(position)
    if not legal:
        return ""

    chosen = engine.search(position, budget_ms(time_left_ms))
    if chosen:
        return chosen
    return bb.move_uci(legal[0])


# ---------------------------------------------------------------------------
# JIT warmup. numba compiles per argument-type signature, so every function the
# search will use is exercised once here, inside the init budget.
# ---------------------------------------------------------------------------

_WARMUP_FENS = (
    "r1bqk2r/pp1pppbp/2n2np1/2p5/2P5/2N1PNP1/PP1P1PBP/R1BQK2R b KQkq - 0 6",
    "8/5pk1/6p1/7p/7P/5PP1/r5K1/3R4 w - - 0 40",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",  # en passant
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # castling
)
_started = time.perf_counter()
for _fen in _WARMUP_FENS:
    _position = bb.from_fen(_fen)
    bb.generate_legal(_position)
    engine.search(_position, 60)
engine.new_game()
print(f"init: warmed in {time.perf_counter() - _started:.1f}s")
