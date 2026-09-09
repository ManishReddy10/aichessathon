"""Bitboard move generation, built to run entirely inside numba.

The search cannot afford python-chess: it is 74% of the engine's runtime and
tops out around 250k nodes/sec. Everything here is arrays and integers so the
whole search can be jitted, with no Python object in the hot loop.

Squares are 0..63 with a1=0 and h8=63, matching python-chess, so results can be
diffed against it. Bitboards are int64: numba has no unsigned type, and the bit
patterns are identical under two's complement.
"""

import numpy as np
from numba import njit

FULL = (1 << 64) - 1
SIGN = 1 << 63


def to_signed(value: int) -> int:
    """Reinterpret an unsigned 64-bit pattern as the int64 numba will hold."""
    value &= FULL
    return value - (1 << 64) if value & SIGN else value


def to_unsigned(value: int) -> int:
    """Reinterpret an int64 back as the unsigned pattern the tests compare."""
    return value & FULL


# ---------------------------------------------------------------------------
# Reference attack generation. Slow, obvious, and only run at import to fill
# the lookup tables — never on the clock.
# ---------------------------------------------------------------------------

KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
ROOK_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
BISHOP_DIRECTIONS = ((1, 1), (1, -1), (-1, -1), (-1, 1))


def _on_board(file: int, rank: int) -> bool:
    return 0 <= file <= 7 and 0 <= rank <= 7


def _leaper_attacks(square: int, deltas: tuple[tuple[int, int], ...]) -> int:
    file, rank = square % 8, square // 8
    mask = 0
    for df, dr in deltas:
        nf, nr = file + df, rank + dr
        if _on_board(nf, nr):
            mask |= 1 << (nr * 8 + nf)
    return mask


def _pawn_attacks(white: bool, square: int) -> int:
    file, rank = square % 8, square // 8
    step = 1 if white else -1
    mask = 0
    for df in (-1, 1):
        nf, nr = file + df, rank + step
        if _on_board(nf, nr):
            mask |= 1 << (nr * 8 + nf)
    return mask


def _ray_attacks(square: int, occupied: int, directions: tuple[tuple[int, int], ...]) -> int:
    """Walk each direction until a blocker, including the blocker's square."""
    file, rank = square % 8, square // 8
    mask = 0
    for df, dr in directions:
        nf, nr = file + df, rank + dr
        while _on_board(nf, nr):
            target = nr * 8 + nf
            mask |= 1 << target
            if occupied & (1 << target):
                break
            nf, nr = nf + df, nr + dr
    return mask


def _relevant_mask(square: int, directions: tuple[tuple[int, int], ...]) -> int:
    """Occupancy squares that can block this piece, excluding the board edge.

    The edge never changes the answer: a blocker there is the last square
    either way, so leaving it out halves the table.
    """
    file, rank = square % 8, square // 8
    mask = 0
    for df, dr in directions:
        nf, nr = file + df, rank + dr
        while _on_board(nf + df, nr + dr):
            mask |= 1 << (nr * 8 + nf)
            nf, nr = nf + df, nr + dr
    return mask


def _subsets(mask: int) -> list[int]:
    """Every subset of the set bits in mask (Carry-Rippler)."""
    found, subset = [], 0
    while True:
        found.append(subset)
        subset = (subset - mask) & mask
        if subset == 0:
            break
    return found


# ---------------------------------------------------------------------------
# Magic numbers. Found once here with a fixed seed so the tables are identical
# on every run, then baked into flat arrays the jitted code indexes.
# ---------------------------------------------------------------------------


def _find_magic(
    square: int, directions: tuple[tuple[int, int], ...], rng: np.random.Generator
) -> tuple[int, int]:
    mask = _relevant_mask(square, directions)
    bits = bin(mask).count("1")
    occupancies = _subsets(mask)
    attacks = [_ray_attacks(square, occ, directions) for occ in occupancies]
    size = 1 << bits

    while True:
        # Sparse magics collide far less often than uniform ones.
        magic = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        magic &= int(rng.integers(0, 1 << 64, dtype=np.uint64))
        magic &= int(rng.integers(0, 1 << 64, dtype=np.uint64))
        if bin((mask * magic) & 0xFF00000000000000).count("1") < 6:
            continue
        table: list[int | None] = [None] * size
        for occupancy, attack in zip(occupancies, attacks, strict=True):
            index = ((occupancy * magic) & FULL) >> (64 - bits)
            if table[index] is None:
                table[index] = attack
            elif table[index] != attack:
                break
        else:
            return magic, bits


def find_magics(directions: tuple[tuple[int, int], ...], seed: int) -> list[int]:
    """Search for a magic per square. Not called at import — see the note below.

    Searching takes ~28s, which is most of the 90s init budget the platform
    allows before the clock starts. The constants below are what this returned
    for the recorded seeds; re-run it to regenerate them.
    """
    rng = np.random.default_rng(seed)
    return [_find_magic(square, directions, rng)[0] for square in range(64)]


_ROOK_MAGIC_CONSTANTS = (
    0x0080008020400010, 0x0840002008100040, 0x2100110020000840, 0x22000A0010044020,
    0x4600020060043008, 0x0180010200800400, 0x5200008801020004, 0x7080038000644100,
    0x00208006882A4000, 0x0000402010004000, 0x0010801000882000, 0x4040800804801002,
    0x0015000800041100, 0x0002000200100408, 0x0114000104100208, 0x8146000204004081,
    0x2000208000804000, 0x0260008020400088, 0x0400808010002008, 0x0050808008001000,
    0x0508004004004200, 0xB483010004000802, 0x0002808082000100, 0x10002200058C1041,
    0x8000400180008120, 0x1010810100204000, 0x0300408600221200, 0x0028100080080081,
    0x0040100500080100, 0x0001200801401004, 0x0010214400089012, 0x0001004200039421,
    0x0080002000400040, 0x2040100060A00800, 0x005D002001004011, 0x8968100009002100,
    0x0815001005000800, 0x0018800400800200, 0x0005001431002200, 0x88806900AA000044,
    0x2088401880208000, 0x0000402010014000, 0x2050002000108080, 0x0000220008420011,
    0x0004000408008080, 0x0002040002008080, 0x030022D008040001, 0x8200148C44020001,
    0x0090422108800100, 0x2820210080400100, 0x0000401220070100, 0x0D00090010002300,
    0x0010080004008080, 0x0004002010080401, 0x80A810086A810400, 0x0040238704104200,
    0x00038001C0221901, 0x0C40420020850216, 0x075540220080110A, 0x20000420F0001901,
    0x1103000800041033, 0x0102001044018802, 0x0801008201300824, 0x0022010080440022,
)

_BISHOP_MAGIC_CONSTANTS = (
    0x0002021004008080, 0x80288108088F0020, 0x00100C00B020308C, 0x80980A4900000000,
    0x0021104000000410, 0x0101100804000028, 0x0000820820040600, 0xA200410088014004,
    0x004110A002240841, 0x001020A602054101, 0x0040214901020000, 0x0830282080200000,
    0x0100111040002400, 0x240000882008010C, 0x0400040C0422082A, 0x0C08028441182100,
    0x4419006088050800, 0x0004241044882040, 0x0024100204001200, 0x0108009412102040,
    0x0061000820081008, 0x0080209110101000, 0x0081000200900403, 0x0022080501008230,
    0x400420004302440D, 0x0004108020410922, 0x8010500041010200, 0x10010400C0440080,
    0x0049010008104010, 0x0003060081006100, 0x0808404021091804, 0x0211024800221800,
    0x6001044000200802, 0x2484100500020400, 0x501C020100020404, 0x8C00208020680200,
    0x0004200200602080, 0x2001022280180800, 0x4288008081010800, 0x0842040040802208,
    0x8000901011000808, 0x0084420820420410, 0x100100804C421001, 0xA020022011100800,
    0x1004012011002A01, 0x0408010802000020, 0x0020014202000090, 0x0010144080200080,
    0x86010410222A0000, 0x2012008421083100, 0x00002204840480A0, 0x0406000108480100,
    0x0010024084884210, 0x0C802104A10A0400, 0x0008081000920000, 0x44A00220A2088002,
    0x2085004044044000, 0x800002088C010901, 0x0000100201008840, 0x1004000101040910,
    0x0100016141050502, 0x00083020C2221205, 0x0090A0A011223081, 0x40201410808C0483,
)


def _build_magic_tables(
    directions: tuple[tuple[int, int], ...], constants: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = np.zeros(64, dtype=np.uint64)
    magics = np.zeros(64, dtype=np.uint64)
    shifts = np.zeros(64, dtype=np.uint64)
    offsets = np.zeros(64, dtype=np.uint64)
    table: list[int] = []

    for square in range(64):
        mask = _relevant_mask(square, directions)
        magic = constants[square]
        bits = bin(mask).count("1")
        masks[square] = mask
        magics[square] = magic
        shifts[square] = 64 - bits
        offsets[square] = len(table)

        entries: list[int] = [0] * (1 << bits)
        for occupancy in _subsets(mask):
            index = ((occupancy * magic) & FULL) >> (64 - bits)
            entries[index] = _ray_attacks(square, occupancy, directions)
        table.extend(entries)

    return masks, magics, shifts, offsets, np.array(table, dtype=np.uint64)


# ---------------------------------------------------------------------------
# The tables themselves, built at import inside the 90s budget.
# ---------------------------------------------------------------------------

KNIGHT_TABLE = np.array(
    [_leaper_attacks(sq, KNIGHT_DELTAS) for sq in range(64)], dtype=np.uint64
)
KING_TABLE = np.array(
    [_leaper_attacks(sq, KING_DELTAS) for sq in range(64)], dtype=np.uint64
)
PAWN_TABLE = np.array(
    [[_pawn_attacks(False, sq) for sq in range(64)],
     [_pawn_attacks(True, sq) for sq in range(64)]],
    dtype=np.uint64,
)

ROOK_MASKS, ROOK_MAGICS, ROOK_SHIFTS, ROOK_OFFSETS, ROOK_TABLE = _build_magic_tables(
    ROOK_DIRECTIONS, _ROOK_MAGIC_CONSTANTS
)
BISHOP_MASKS, BISHOP_MAGICS, BISHOP_SHIFTS, BISHOP_OFFSETS, BISHOP_TABLE = _build_magic_tables(
    BISHOP_DIRECTIONS, _BISHOP_MAGIC_CONSTANTS
)


# ---------------------------------------------------------------------------
# Lookups. These are the shapes the jitted search will call.
# ---------------------------------------------------------------------------


def knight_attacks(square: int) -> int:
    return int(KNIGHT_TABLE[square])


def king_attacks(square: int) -> int:
    return int(KING_TABLE[square])


def pawn_attacks(white: bool, square: int) -> int:
    return int(PAWN_TABLE[1 if white else 0][square])


def _magic_lookup(
    square: int, occupied: int, masks: np.ndarray, magics: np.ndarray,
    shifts: np.ndarray, offsets: np.ndarray, table: np.ndarray,
) -> int:
    relevant = occupied & int(masks[square])
    index = ((relevant * int(magics[square])) & FULL) >> int(shifts[square])
    return int(table[int(offsets[square]) + index])


def rook_attacks(square: int, occupied: int) -> int:
    return _magic_lookup(
        square, occupied, ROOK_MASKS, ROOK_MAGICS, ROOK_SHIFTS, ROOK_OFFSETS, ROOK_TABLE
    )


def bishop_attacks(square: int, occupied: int) -> int:
    return _magic_lookup(
        square, occupied, BISHOP_MASKS, BISHOP_MAGICS, BISHOP_SHIFTS, BISHOP_OFFSETS, BISHOP_TABLE
    )


def queen_attacks(square: int, occupied: int) -> int:
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)


# ---------------------------------------------------------------------------
# Position. Two flat arrays so the whole thing can cross into numba and be
# copied cheaply: bitboards, then the scalars that are not bitboards.
# ---------------------------------------------------------------------------

# bbs[0:6]   white pawn, knight, bishop, rook, queen, king
# bbs[6:12]  the same for black
# bbs[12]    white occupancy, bbs[13] black occupancy, bbs[14] both
WHITE_OFFSET, BLACK_OFFSET = 0, 6
OCC_WHITE, OCC_BLACK, OCC_ALL = 12, 13, 14
BB_LEN = 15

# state[0] side to move (0 white, 1 black), [1] castling bits,
# state[2] en passant square or -1, [3] halfmove clock
SIDE, CASTLING, EP, HALFMOVE = 0, 1, 2, 3
STATE_LEN = 4

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

Position = tuple[np.ndarray, np.ndarray]

_PIECE_LETTERS = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def from_fen(fen: str) -> Position:
    """Parse a FEN into the two arrays the jitted code works on."""
    bbs = np.zeros(BB_LEN, dtype=np.uint64)
    state = np.zeros(STATE_LEN, dtype=np.int64)
    placement, side, castling, ep, halfmove = [*fen.split(), "-", "-", "0", "1"][:5]

    square = 56  # FEN starts at a8
    for character in placement:
        if character == "/":
            square -= 16
        elif character.isdigit():
            square += int(character)
        else:
            index = _PIECE_LETTERS[character.lower()]
            index += WHITE_OFFSET if character.isupper() else BLACK_OFFSET
            bbs[index] |= np.uint64(1) << np.uint64(square)
            square += 1

    for index in range(6):
        bbs[OCC_WHITE] |= bbs[WHITE_OFFSET + index]
        bbs[OCC_BLACK] |= bbs[BLACK_OFFSET + index]
    bbs[OCC_ALL] = bbs[OCC_WHITE] | bbs[OCC_BLACK]

    state[SIDE] = 0 if side == "w" else 1
    rights = 0
    for character, bit in (("K", CASTLE_WK), ("Q", CASTLE_WQ), ("k", CASTLE_BK), ("q", CASTLE_BQ)):
        if character in castling:
            rights |= bit
    state[CASTLING] = rights
    state[EP] = -1 if ep == "-" else (ord(ep[0]) - ord("a")) + 8 * (int(ep[1]) - 1)
    state[HALFMOVE] = int(halfmove)
    return bbs, state


def piece_bitboard(position: Position, white: bool, piece_type: int) -> int:
    """piece_type follows python-chess: PAWN=1 .. KING=6."""
    bbs, _ = position
    return int(bbs[(WHITE_OFFSET if white else BLACK_OFFSET) + piece_type - 1])


def occupied(position: Position) -> int:
    return int(position[0][OCC_ALL])


def white_to_move(position: Position) -> bool:
    return bool(position[1][SIDE] == 0)


def en_passant_square(position: Position) -> int:
    return int(position[1][EP])


def halfmove_clock(position: Position) -> int:
    return int(position[1][HALFMOVE])


def can_castle(position: Position, colour: bool, kingside: bool) -> bool:
    white_bit = CASTLE_WK if kingside else CASTLE_WQ
    black_bit = CASTLE_BK if kingside else CASTLE_BQ
    bit = white_bit if colour else black_bit  # chess.WHITE is True
    return bool(int(position[1][CASTLING]) & bit)


def is_attacked(position: Position, square: int, by_white: bool) -> bool:
    """Is `square` attacked by the given colour, ignoring whose turn it is."""
    bbs, _ = position
    offset = WHITE_OFFSET if by_white else BLACK_OFFSET
    occ = int(bbs[OCC_ALL])

    # A pawn attacks `square` exactly when `square` attacks it as the other colour.
    if int(PAWN_TABLE[0 if by_white else 1][square]) & int(bbs[offset + 0]):
        return True
    if int(KNIGHT_TABLE[square]) & int(bbs[offset + 1]):
        return True
    if int(KING_TABLE[square]) & int(bbs[offset + 5]):
        return True
    if rook_attacks(square, occ) & (int(bbs[offset + 3]) | int(bbs[offset + 4])):
        return True
    return bool(bishop_attacks(square, occ) & (int(bbs[offset + 2]) | int(bbs[offset + 4])))


# ---------------------------------------------------------------------------
# Moves. Packed into one int so a move list is an array, not a list of objects.
# bits 0-5 from, 6-11 to, 12-14 promotion, 15-16 flag
# ---------------------------------------------------------------------------

FLAG_NORMAL, FLAG_EP, FLAG_CASTLE, FLAG_DOUBLE = 0, 1, 2, 3
PROMO_LETTERS = ("", "n", "b", "r", "q")


def encode(from_square: int, to_square: int, promotion: int = 0, flag: int = 0) -> int:
    return from_square | (to_square << 6) | (promotion << 12) | (flag << 15)


def move_from(move: int) -> int:
    return move & 63


def move_to(move: int) -> int:
    return (move >> 6) & 63


def move_promotion(move: int) -> int:
    return (move >> 12) & 7


def move_flag(move: int) -> int:
    return (move >> 15) & 3


def move_uci(move: int) -> str:
    def name(square: int) -> str:
        return chr(ord("a") + square % 8) + str(square // 8 + 1)

    return name(move_from(move)) + name(move_to(move)) + PROMO_LETTERS[move_promotion(move)]


def _bits(mask: int) -> list[int]:
    """Indices of the set bits, low to high."""
    found = []
    while mask:
        low = mask & -mask
        found.append(low.bit_length() - 1)
        mask ^= low
    return found


def _unpack(position: Position) -> tuple[list[int], list[int]]:
    return [int(x) for x in position[0]], [int(x) for x in position[1]]


def _pack(boards: list[int], state: list[int]) -> Position:
    return np.array(boards, dtype=np.uint64), np.array(state, dtype=np.int64)


RANK_2, RANK_7 = 0x000000000000FF00, 0x00FF000000000000
PROMOTION_RANKS = 0xFF000000000000FF


def generate_pseudo_legal(position: Position) -> list[int]:
    boards, state = _unpack(position)
    white = state[SIDE] == 0
    us = WHITE_OFFSET if white else BLACK_OFFSET
    own = boards[OCC_WHITE if white else OCC_BLACK]
    enemy = boards[OCC_BLACK if white else OCC_WHITE]
    occ = boards[OCC_ALL]
    moves: list[int] = []

    # --- pawns ---
    forward = 8 if white else -8
    start_rank = RANK_2 if white else RANK_7
    for square in _bits(boards[us + 0]):
        target = square + forward
        if not occ & (1 << target):
            if (1 << target) & PROMOTION_RANKS:
                moves.extend(encode(square, target, promo) for promo in (4, 3, 2, 1))
            else:
                moves.append(encode(square, target))
                double = target + forward
                if (1 << square) & start_rank and not occ & (1 << double):
                    moves.append(encode(square, double, 0, FLAG_DOUBLE))
        for capture in _bits(int(PAWN_TABLE[1 if white else 0][square]) & enemy):
            if (1 << capture) & PROMOTION_RANKS:
                moves.extend(encode(square, capture, promo) for promo in (4, 3, 2, 1))
            else:
                moves.append(encode(square, capture))
        if state[EP] >= 0 and int(PAWN_TABLE[1 if white else 0][square]) & (1 << state[EP]):
            moves.append(encode(square, state[EP], 0, FLAG_EP))

    # --- knights and king ---
    for square in _bits(boards[us + 1]):
        for target in _bits(int(KNIGHT_TABLE[square]) & ~own):
            moves.append(encode(square, target))
    for square in _bits(boards[us + 5]):
        for target in _bits(int(KING_TABLE[square]) & ~own):
            moves.append(encode(square, target))

    # --- sliders ---
    for square in _bits(boards[us + 2]):
        for target in _bits(bishop_attacks(square, occ) & ~own):
            moves.append(encode(square, target))
    for square in _bits(boards[us + 3]):
        for target in _bits(rook_attacks(square, occ) & ~own):
            moves.append(encode(square, target))
    for square in _bits(boards[us + 4]):
        for target in _bits(queen_attacks(square, occ) & ~own):
            moves.append(encode(square, target))

    # --- castling: rights, a clear path, and no attacked square on the way ---
    king_square = 4 if white else 60
    if boards[us + 5] & (1 << king_square) and not is_attacked(position, king_square, not white):
        kingside = CASTLE_WK if white else CASTLE_BK
        queenside = CASTLE_WQ if white else CASTLE_BQ
        shift = 0 if white else 56
        if (state[CASTLING] & kingside and not occ & (0x60 << shift)
                and not any(is_attacked(position, king_square + s, not white) for s in (1, 2))):
            moves.append(encode(king_square, king_square + 2, 0, FLAG_CASTLE))
        if (state[CASTLING] & queenside and not occ & (0x0E << shift)
                and not any(is_attacked(position, king_square - s, not white) for s in (1, 2))):
            moves.append(encode(king_square, king_square - 2, 0, FLAG_CASTLE))

    return moves


_CASTLE_RIGHT_BY_SQUARE = {0: CASTLE_WQ, 7: CASTLE_WK, 56: CASTLE_BQ, 63: CASTLE_BK}


def make(position: Position, move: int) -> Position:
    boards, state = _unpack(position)
    white = state[SIDE] == 0
    us = WHITE_OFFSET if white else BLACK_OFFSET
    them = BLACK_OFFSET if white else WHITE_OFFSET
    origin, target = move_from(move), move_to(move)
    flag, promotion = move_flag(move), move_promotion(move)

    piece = next(i for i in range(6) if boards[us + i] & (1 << origin))
    boards[us + piece] ^= (1 << origin) | (1 << target)

    captured = None
    if flag == FLAG_EP:
        victim = target - 8 if white else target + 8
        boards[them + 0] ^= 1 << victim
        captured = 0
    else:
        for index in range(6):
            if boards[them + index] & (1 << target):
                boards[them + index] ^= 1 << target
                captured = index
                break

    if promotion:
        boards[us + 0] ^= 1 << target
        boards[us + promotion] |= 1 << target

    if flag == FLAG_CASTLE:
        if target > origin:
            rook_from, rook_to = target + 1, target - 1
        else:
            rook_from, rook_to = target - 2, target + 1
        boards[us + 3] ^= (1 << rook_from) | (1 << rook_to)

    # Castling rights die when the king moves, or when a rook leaves or is taken.
    rights = state[CASTLING]
    if piece == 5:
        rights &= ~(CASTLE_WK | CASTLE_WQ) if white else ~(CASTLE_BK | CASTLE_BQ)
    rights &= ~_CASTLE_RIGHT_BY_SQUARE.get(origin, 0)
    rights &= ~_CASTLE_RIGHT_BY_SQUARE.get(target, 0)

    state[CASTLING] = rights
    state[EP] = (origin + target) // 2 if flag == FLAG_DOUBLE else -1
    state[HALFMOVE] = 0 if (piece == 0 or captured is not None) else state[HALFMOVE] + 1
    state[SIDE] = 1 - state[SIDE]

    boards[OCC_WHITE] = 0
    boards[OCC_BLACK] = 0
    for index in range(6):
        boards[OCC_WHITE] |= boards[WHITE_OFFSET + index]
        boards[OCC_BLACK] |= boards[BLACK_OFFSET + index]
    boards[OCC_ALL] = boards[OCC_WHITE] | boards[OCC_BLACK]
    return _pack(boards, state)


def in_check(position: Position, white: bool) -> bool:
    boards = position[0]
    king = int(boards[(WHITE_OFFSET if white else BLACK_OFFSET) + 5])
    if not king:
        return False
    return is_attacked(position, king.bit_length() - 1, not white)


def generate_legal(position: Position) -> list[int]:
    mover_is_white = white_to_move(position)
    legal = []
    for move in generate_pseudo_legal(position):
        if not in_check(make(position, move), mover_is_white):
            legal.append(move)
    return legal


def legal_move_ucis(position: Position) -> list[str]:
    return [move_uci(move) for move in generate_legal(position)]


def perft(position: Position, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return len(generate_legal(position))
    return sum(perft(make(position, move), depth - 1) for move in generate_legal(position))


# ---------------------------------------------------------------------------
# The jitted path. Same logic as above, but with nothing numba cannot compile:
# no Python lists, no tuples of arrays, no boxing. The functions above stay as
# the readable reference the tests check this against.
# ---------------------------------------------------------------------------

ONE = np.uint64(1)
ZERO = np.uint64(0)
ONES = np.uint64(0xFFFFFFFFFFFFFFFF)

# De Bruijn sequence for finding the index of the lowest set bit. numba has no
# bit_length on uint64, and testing 64 squares one at a time is far slower.
_DEBRUIJN = 0x03F79D71B4CB0A89
_DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _DEBRUIJN_INDEX[((_DEBRUIJN << _i) & FULL) >> 58] = _i
DEBRUIJN = np.uint64(_DEBRUIJN)
DEBRUIJN_INDEX = _DEBRUIJN_INDEX

MAX_MOVES = 256


@njit(cache=False)
def lsb_index(mask):
    """Index of the lowest set bit. Undefined for zero, so callers check first."""
    isolated = mask & ((mask ^ ONES) + ONE)
    return DEBRUIJN_INDEX[(isolated * DEBRUIJN) >> np.uint64(58)]


@njit(cache=False)
def rook_attacks_jit(square, occ):
    relevant = occ & ROOK_MASKS[square]
    index = (relevant * ROOK_MAGICS[square]) >> ROOK_SHIFTS[square]
    return ROOK_TABLE[ROOK_OFFSETS[square] + np.int64(index)]


@njit(cache=False)
def bishop_attacks_jit(square, occ):
    relevant = occ & BISHOP_MASKS[square]
    index = (relevant * BISHOP_MAGICS[square]) >> BISHOP_SHIFTS[square]
    return BISHOP_TABLE[BISHOP_OFFSETS[square] + np.int64(index)]


@njit(cache=False)
def is_attacked_jit(bbs, square, by_white):
    offset = 0 if by_white else 6
    occ = bbs[OCC_ALL]
    if PAWN_TABLE[0 if by_white else 1][square] & bbs[offset + 0]:
        return True
    if KNIGHT_TABLE[square] & bbs[offset + 1]:
        return True
    if KING_TABLE[square] & bbs[offset + 5]:
        return True
    if rook_attacks_jit(square, occ) & (bbs[offset + 3] | bbs[offset + 4]):
        return True
    return bool(bishop_attacks_jit(square, occ) & (bbs[offset + 2] | bbs[offset + 4]))


@njit(cache=False)
def generate_jit(bbs, state, moves):
    """Fill `moves` with pseudo-legal moves and return how many there are."""
    white = state[SIDE] == 0
    us = 0 if white else 6
    own = bbs[OCC_WHITE] if white else bbs[OCC_BLACK]
    enemy = bbs[OCC_BLACK] if white else bbs[OCC_WHITE]
    occ = bbs[OCC_ALL]
    not_own = own ^ ONES
    count = 0

    forward = 8 if white else -8
    start_rank = np.uint64(0x000000000000FF00) if white else np.uint64(0x00FF000000000000)
    promo_ranks = np.uint64(0xFF000000000000FF)

    pawns = bbs[us + 0]
    while pawns:
        square = lsb_index(pawns)
        pawns &= pawns - ONE
        target = square + forward
        if not (occ >> np.uint64(target)) & ONE:
            if (ONE << np.uint64(target)) & promo_ranks:
                for promo in (4, 3, 2, 1):
                    moves[count] = square | (target << 6) | (promo << 12)
                    count += 1
            else:
                moves[count] = square | (target << 6)
                count += 1
                double = target + forward
                if ((ONE << np.uint64(square)) & start_rank) and not (
                    (occ >> np.uint64(double)) & ONE
                ):
                    moves[count] = square | (double << 6) | (FLAG_DOUBLE << 15)
                    count += 1
        captures = PAWN_TABLE[1 if white else 0][square] & enemy
        while captures:
            capture = lsb_index(captures)
            captures &= captures - ONE
            if (ONE << np.uint64(capture)) & promo_ranks:
                for promo in (4, 3, 2, 1):
                    moves[count] = square | (capture << 6) | (promo << 12)
                    count += 1
            else:
                moves[count] = square | (capture << 6)
                count += 1
        if state[EP] >= 0 and (PAWN_TABLE[1 if white else 0][square] >> np.uint64(state[EP])) & ONE:
            moves[count] = square | (state[EP] << 6) | (FLAG_EP << 15)
            count += 1

    for piece, is_slider in ((1, 0), (5, 0), (2, 1), (3, 2), (4, 3)):
        pieces = bbs[us + piece]
        while pieces:
            square = lsb_index(pieces)
            pieces &= pieces - ONE
            if is_slider == 0:
                attacks = KNIGHT_TABLE[square] if piece == 1 else KING_TABLE[square]
            elif is_slider == 1:
                attacks = bishop_attacks_jit(square, occ)
            elif is_slider == 2:
                attacks = rook_attacks_jit(square, occ)
            else:
                attacks = rook_attacks_jit(square, occ) | bishop_attacks_jit(square, occ)
            attacks &= not_own
            while attacks:
                target = lsb_index(attacks)
                attacks &= attacks - ONE
                moves[count] = square | (target << 6)
                count += 1

    king_square = 4 if white else 60
    if (bbs[us + 5] >> np.uint64(king_square)) & ONE and not is_attacked_jit(
        bbs, king_square, not white
    ):
        shift = np.uint64(0) if white else np.uint64(56)
        kingside = CASTLE_WK if white else CASTLE_BK
        queenside = CASTLE_WQ if white else CASTLE_BQ
        if (state[CASTLING] & kingside and not occ & (np.uint64(0x60) << shift)
                and not is_attacked_jit(bbs, king_square + 1, not white)
                and not is_attacked_jit(bbs, king_square + 2, not white)):
            moves[count] = king_square | ((king_square + 2) << 6) | (FLAG_CASTLE << 15)
            count += 1
        if (state[CASTLING] & queenside and not occ & (np.uint64(0x0E) << shift)
                and not is_attacked_jit(bbs, king_square - 1, not white)
                and not is_attacked_jit(bbs, king_square - 2, not white)):
            moves[count] = king_square | ((king_square - 2) << 6) | (FLAG_CASTLE << 15)
            count += 1

    return count


@njit(cache=False)
def castle_rights_lost(square):
    if square == 0:
        return CASTLE_WQ
    if square == 7:
        return CASTLE_WK
    if square == 56:
        return CASTLE_BQ
    if square == 63:
        return CASTLE_BK
    return 0


@njit(cache=False)
def make_jit(bbs, state, move, undo):
    """Apply `move` in place, recording in `undo` what unmake_jit needs."""
    white = state[SIDE] == 0
    us = 0 if white else 6
    them = 6 if white else 0
    origin = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    flag = (move >> 15) & 3

    piece = 0
    for index in range(6):
        if (bbs[us + index] >> np.uint64(origin)) & ONE:
            piece = index
            break

    undo[1] = state[CASTLING]
    undo[2] = state[EP]
    undo[3] = state[HALFMOVE]
    undo[4] = piece

    bbs[us + piece] ^= (ONE << np.uint64(origin)) | (ONE << np.uint64(target))

    captured = -1
    if flag == FLAG_EP:
        victim = target - 8 if white else target + 8
        bbs[them + 0] ^= ONE << np.uint64(victim)
        captured = 0
    else:
        for index in range(6):
            if (bbs[them + index] >> np.uint64(target)) & ONE:
                bbs[them + index] ^= ONE << np.uint64(target)
                captured = index
                break
    undo[0] = captured

    if promotion:
        bbs[us + 0] ^= ONE << np.uint64(target)
        bbs[us + promotion] |= ONE << np.uint64(target)

    if flag == FLAG_CASTLE:
        if target > origin:
            rook_from, rook_to = target + 1, target - 1
        else:
            rook_from, rook_to = target - 2, target + 1
        bbs[us + 3] ^= (ONE << np.uint64(rook_from)) | (ONE << np.uint64(rook_to))

    rights = state[CASTLING]
    if piece == 5:
        rights &= ~(CASTLE_WK | CASTLE_WQ) if white else ~(CASTLE_BK | CASTLE_BQ)
    rights &= ~castle_rights_lost(origin)
    rights &= ~castle_rights_lost(target)
    state[CASTLING] = rights
    state[EP] = (origin + target) // 2 if flag == FLAG_DOUBLE else -1
    state[HALFMOVE] = 0 if (piece == 0 or captured >= 0) else state[HALFMOVE] + 1
    state[SIDE] = 1 - state[SIDE]

    white_occ = ZERO
    black_occ = ZERO
    for index in range(6):
        white_occ |= bbs[index]
        black_occ |= bbs[6 + index]
    bbs[OCC_WHITE] = white_occ
    bbs[OCC_BLACK] = black_occ
    bbs[OCC_ALL] = white_occ | black_occ


@njit(cache=False)
def unmake_jit(bbs, state, move, undo):
    state[SIDE] = 1 - state[SIDE]
    white = state[SIDE] == 0
    us = 0 if white else 6
    them = 6 if white else 0
    origin = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    flag = (move >> 15) & 3
    captured = undo[0]
    piece = undo[4]

    if promotion:
        bbs[us + promotion] ^= ONE << np.uint64(target)
        bbs[us + 0] |= ONE << np.uint64(target)

    bbs[us + piece] ^= (ONE << np.uint64(origin)) | (ONE << np.uint64(target))

    if flag == FLAG_EP:
        victim = target - 8 if white else target + 8
        bbs[them + 0] |= ONE << np.uint64(victim)
    elif captured >= 0:
        bbs[them + captured] |= ONE << np.uint64(target)

    if flag == FLAG_CASTLE:
        if target > origin:
            rook_from, rook_to = target + 1, target - 1
        else:
            rook_from, rook_to = target - 2, target + 1
        bbs[us + 3] ^= (ONE << np.uint64(rook_from)) | (ONE << np.uint64(rook_to))

    state[CASTLING] = undo[1]
    state[EP] = undo[2]
    state[HALFMOVE] = undo[3]

    white_occ = ZERO
    black_occ = ZERO
    for index in range(6):
        white_occ |= bbs[index]
        black_occ |= bbs[6 + index]
    bbs[OCC_WHITE] = white_occ
    bbs[OCC_BLACK] = black_occ
    bbs[OCC_ALL] = white_occ | black_occ


@njit(cache=False)
def perft_jit(bbs, state, depth, stack, undo_stack):
    if depth == 0:
        return 1
    moves = stack[depth]
    undo = undo_stack[depth]
    count = generate_jit(bbs, state, moves)
    mover_white = state[SIDE] == 0
    total = 0
    for index in range(count):
        move = moves[index]
        make_jit(bbs, state, move, undo)
        king = bbs[(0 if mover_white else 6) + 5]
        if king and not is_attacked_jit(bbs, lsb_index(king), not mover_white):
            total += perft_jit(bbs, state, depth - 1, stack, undo_stack)
        unmake_jit(bbs, state, move, undo)
    return total


def _fresh_stacks(depth: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((depth + 2, MAX_MOVES), dtype=np.int64),
        np.zeros((depth + 2, 5), dtype=np.int64),
    )


def perft_fast(position: Position, depth: int) -> int:
    bbs, state = position[0].copy(), position[1].copy()
    stack, undo_stack = _fresh_stacks(depth)
    return int(perft_jit(bbs, state, depth, stack, undo_stack))


def legal_move_ucis_fast(position: Position) -> list[str]:
    bbs, state = position[0].copy(), position[1].copy()
    moves = np.zeros(MAX_MOVES, dtype=np.int64)
    undo = np.zeros(5, dtype=np.int64)
    count = generate_jit(bbs, state, moves)
    mover_white = state[SIDE] == 0
    found = []
    for index in range(count):
        move = int(moves[index])
        make_jit(bbs, state, move, undo)
        king = bbs[(0 if mover_white else 6) + 5]
        if king and not is_attacked_jit(bbs, lsb_index(king), not mover_white):
            found.append(move_uci(move))
        unmake_jit(bbs, state, move, undo)
    return found
