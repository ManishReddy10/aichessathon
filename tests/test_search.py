"""The platform's contract: a FEN in, a legal UCI move out, inside the clock.

These test agent.get_move end to end, whatever search sits behind it. Every
expected answer is derived from python-chess at test time rather than hardcoded,
so a failure means the engine is wrong, not that a FEN was transcribed badly.
The search internals are covered in test_engine.py.
"""

import time

import chess
import pytest

import agent

OPEN_MIDDLEGAME = "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 11"


def forcing_mates(fen: str) -> dict[str, int]:
    """Map each move that forces mate to how many white moves it takes (1 or 2)."""
    board = chess.Board(fen)
    found: dict[str, int] = {}
    for first in board.legal_moves:
        board.push(first)
        if board.is_checkmate():
            found[first.uci()] = 1
        elif list(board.legal_moves) and _every_reply_is_mated(board):
            found[first.uci()] = 2
        board.pop()
    return found


def _every_reply_is_mated(board: chess.Board) -> bool:
    for reply in board.legal_moves:
        board.push(reply)
        mated = any(_mates(board, follow_up) for follow_up in board.legal_moves)
        board.pop()
        if not mated:
            return False
    return True


def _mates(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    result = board.is_checkmate()
    board.pop()
    return result


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    agent.new_game()


# --- mates ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",  # back rank, Ra8#
        "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",  # smothered, Nf7#
    ],
)
def test_plays_an_available_mate_in_one(fen: str) -> None:
    mates = forcing_mates(fen)
    assert mates, "test position is wrong: no forcing mate exists"
    assert agent.get_move(fen, 5_000) in [m for m, plies in mates.items() if plies == 1]


def test_prefers_a_mate_in_one_over_a_mate_in_two() -> None:
    """Mate scores must shorten with ply, or the engine dawdles in a won position."""
    fen = "7k/8/8/8/8/8/6R1/6RK w - - 0 1"
    mates = forcing_mates(fen)
    quickest = [move for move, plies in mates.items() if plies == 1]
    assert quickest and len(mates) > len(quickest), "test position is wrong"
    assert agent.get_move(fen, 5_000) in quickest


def test_finds_a_forcing_mate_in_two() -> None:
    fen = "7k/8/8/8/8/8/R7/1R6 w - - 0 1"
    mates = forcing_mates(fen)
    assert mates and 1 not in mates.values(), "test position is wrong: a mate in one exists"
    assert agent.get_move(fen, 8_000) in mates


def test_returns_empty_when_already_checkmated() -> None:
    """No legal moves: get_move must return rather than raise or index into nothing."""
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # fool's mate
    assert chess.Board(fen).is_checkmate(), "test position is wrong: not checkmate"
    assert agent.get_move(fen, 1_000) == ""


# --- legality and the clock -----------------------------------------------


@pytest.mark.parametrize("budget_ms", [30, 60, 250, 2_000])
def test_returns_a_legal_move_on_any_budget(budget_ms: int) -> None:
    board = chess.Board(OPEN_MIDDLEGAME)
    assert chess.Move.from_uci(agent.get_move(OPEN_MIDDLEGAME, budget_ms)) in board.legal_moves


@pytest.mark.parametrize(
    "fen",
    [
        "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1",  # in check
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",  # en passant available
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # castling available
        "8/P6k/8/8/8/8/6K1/8 w - - 0 1",  # promotion forced
    ],
)
def test_returns_a_legal_move_in_awkward_positions(fen: str) -> None:
    board = chess.Board(fen)
    assert chess.Move.from_uci(agent.get_move(fen, 2_000)) in board.legal_moves


def test_promotes_with_a_uci_promotion_suffix() -> None:
    fen = "8/P6k/8/8/8/8/6K1/8 w - - 0 1"
    board = chess.Board(fen)
    move = chess.Move.from_uci(agent.get_move(fen, 2_000))
    assert move in board.legal_moves
    assert move.promotion is not None, "the only sensible move is a promotion"


def test_respects_the_clock_it_was_handed() -> None:
    """Overshooting the allotted budget is a flag, which loses the game outright."""
    started = time.monotonic()
    agent.get_move(OPEN_MIDDLEGAME, 1_000)
    assert time.monotonic() - started < 1.0


def test_budget_never_exceeds_the_clock() -> None:
    for clock in (500, 1_500, 5_000, 20_000, 120_000):
        assert agent.budget_ms(clock) < clock, clock


# --- playing a game through the platform's shape ---------------------------


def test_plays_a_sequence_of_moves_without_repeating_itself() -> None:
    """get_move is handed a bare FEN each time; history has to survive between calls."""
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3QK3 w - - 0 1")  # white a queen up
    for _ in range(12):
        if board.is_game_over() or board.is_repetition(3):
            break
        move = chess.Move.from_uci(agent.get_move(board.fen(), 800))
        assert move in board.legal_moves
        board.push(move)
    assert not board.is_repetition(3), "shuffled into a threefold draw while winning"
