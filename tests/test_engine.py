"""The jitted evaluation and search, built on bitboard.py."""

import chess
import pytest

import bitboard as bb
import engine

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 11",
    "8/5pk1/6p1/7p/7P/5PP1/r5K1/3R4 w - - 0 40",
]


def mirror(fen: str) -> str:
    """Same position with the colours swapped, so evaluation must negate."""
    return chess.Board(fen).mirror().fen()


# --- zobrist --------------------------------------------------------------


@pytest.mark.parametrize("fen", FENS)
def test_zobrist_is_stable_for_the_same_position(fen: str) -> None:
    assert engine.zobrist(bb.from_fen(fen)) == engine.zobrist(bb.from_fen(fen))


def test_zobrist_separates_positions_differing_only_in_side_to_move() -> None:
    white = bb.from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    black = bb.from_fen("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    assert engine.zobrist(white) != engine.zobrist(black)


def test_zobrist_separates_positions_differing_only_in_castling_rights() -> None:
    with_rights = bb.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without = bb.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert engine.zobrist(with_rights) != engine.zobrist(without)


@pytest.mark.parametrize("fen", FENS)
def test_zobrist_is_restored_after_make_and_unmake(fen: str) -> None:
    """A key that drifts across make/unmake poisons every transposition lookup."""
    position = bb.from_fen(fen)
    before = engine.zobrist(position)
    assert engine.probe_make_unmake_keeps_zobrist(position), fen
    assert engine.zobrist(position) == before, "zobrist() must not mutate the position"


# --- evaluation -----------------------------------------------------------


@pytest.mark.parametrize("fen", FENS)
def test_evaluation_is_colour_symmetric(fen: str) -> None:
    """Scores are from the mover's side, so mirroring the board must not change them.

    Any asymmetry here is a bug that quietly favours one colour, and the arena
    would show it as a colour split rather than as an obvious failure.
    """
    assert engine.evaluate(bb.from_fen(fen)) == engine.evaluate(bb.from_fen(mirror(fen)))


def test_being_a_queen_up_evaluates_better_than_level() -> None:
    level = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
    a_queen_up = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"))
    assert a_queen_up > level + 800


def test_endgame_king_prefers_the_centre() -> None:
    centred = engine.evaluate(bb.from_fen("8/8/8/8/4K3/8/6P1/7k w - - 0 1"))
    cornered = engine.evaluate(bb.from_fen("8/8/8/8/8/8/6P1/K6k w - - 0 1"))
    assert centred > cornered


# --- search ---------------------------------------------------------------


def forcing_mates(fen: str) -> dict[str, int]:
    board = chess.Board(fen)
    found: dict[str, int] = {}
    for first in board.legal_moves:
        board.push(first)
        if board.is_checkmate():
            found[first.uci()] = 1
        board.pop()
    return found


@pytest.mark.parametrize(
    "fen",
    [
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
        "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",
    ],
)
def test_search_plays_an_available_mate_in_one(fen: str) -> None:
    mates = forcing_mates(fen)
    assert mates, "test position is wrong"
    assert engine.search(bb.from_fen(fen), 2_000) in mates


def test_search_prefers_the_shorter_mate() -> None:
    fen = "7k/8/8/8/8/8/6R1/6RK w - - 0 1"
    mates = forcing_mates(fen)
    assert mates, "test position is wrong"
    assert engine.search(bb.from_fen(fen), 3_000) in mates


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("budget_ms", [40, 300, 1_500])
def test_search_always_returns_a_legal_move(fen: str, budget_ms: int) -> None:
    legal = bb.legal_move_ucis(bb.from_fen(fen))
    assert engine.search(bb.from_fen(fen), budget_ms) in legal


def test_search_respects_its_budget() -> None:
    import time

    started = time.monotonic()
    engine.search(bb.from_fen(FENS[1]), 500)
    assert time.monotonic() - started < 1.2, "overshooting the budget is a flag"


def test_search_finds_a_free_queen() -> None:
    """A one-ply tactic: the queen on d5 is hanging to the pawn on e4."""
    fen = "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"
    assert engine.search(bb.from_fen(fen), 2_000) == "e4d5"


def test_search_is_much_faster_than_the_python_chess_engine() -> None:
    """The whole reason this module exists."""
    nodes = engine.search_nodes(bb.from_fen(FENS[2]), 1_000)
    assert nodes > 300_000, f"only {nodes:,} nodes in 1s -- no faster than python-chess"
