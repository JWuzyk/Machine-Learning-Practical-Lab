import copy
from gzip import GzipFile
from os import path
from .game import TwoPlayerFiniteGame
import pickle


class LGame(TwoPlayerFiniteGame):
    """Represent a board state of the L-Game."""
    __slots__ = ()

    ROTATIONS = (
        # 0
        ("▓▓",
         " ▓",
         " ▓"),

        # 1
        ("▓▓▓",
         "▓"),

        # 2
        ("▓",
         "▓",
         "▓▓"),

        # 3
        ("  ▓",
         "▓▓▓"),

        # 4
        ("▓▓",
         "▓",
         "▓"),

        # 5
        ("▓",
         "▓▓▓"),

        # 6
        (" ▓",
         " ▓",
         "▓▓"),

        # 7
        ("▓▓▓",
         "  ▓"),
    )

    N = ("▗▄▖",
         "▝▀▘")

    def __new__(cls, state=None, **kwargs):
        if state is None:
            for kw in ('n1', 'n2', 'l1', 'l2'):
                val = kwargs.get(kw)
                if isinstance(val, tuple):
                    kwargs[kw] = _pos_to_index(val)
            n1 = kwargs.get('n1', 0)
            n2 = kwargs.get('n2', 15)
            if n2 < n1:
                n1, n2 = n2, n1
            state = (
                kwargs.get('l1', 1),
                kwargs.get('l1_rot', 0),
                kwargs.get('l2', 5),
                kwargs.get('l2_rot', 2),
                n1, n2,
                kwargs.get('player', 1))
        return tuple.__new__(cls, state)

    def valid_actions(self):
        """Get the valid actions for a given player."""
        board = [[False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False]]
        # Mark opponents L-piece
        x, y, rot = (*self.l1, self.l1_rot) if self.player == 2 else (*self.l2, self.l2_rot)
        _mark_l_piece(board, x, y, rot)

        # Mark N-pieces
        for n in (self.n1, self.n2):
            x, y = _pos_to_image(*n)
            board[y][x] = True

        old_pos, old_rot = (self.l1, self.l1_rot) if self.player == 1 else (self.l2, self.l2_rot)
        old_pos = _pos_to_image(*old_pos)
        for x, y, rot in _get_l_piece_moves(board):
            if (x, y) == old_pos and rot == old_rot:
                continue
            board_prime = copy.deepcopy(board)
            _mark_l_piece(board_prime, x, y, rot, image=True)
            index = _image_to_index(x, y)
            for n1_idx, n2_idx in _get_n_piece_moves(board_prime, self.n1_idx, self.n2_idx):
                yield index, rot, n1_idx, n2_idx

    def winner(self):
        """Check if there is a winner."""
        if not self.valid_actions():
            return self.next_player
        return None

    def apply_action(self, action):
        """Apply an action, i.e. a move as returned by `valid_actions()`."""
        index, rot, n1_idx, n2_idx = action
        if n1_idx not in (self.n1_idx, self.n2_idx):
            assert n2_idx in (self.n1_idx, self.n2_idx), "Can move only one N-piece"
        if n2_idx not in (self.n1_idx, self.n2_idx):
            assert n1_idx in (self.n1_idx, self.n2_idx), "Can move only one N-piece"
        new = self._as_dict()
        new['n1'] = min(n1_idx, n2_idx)
        new['n2'] = max(n1_idx, n2_idx)
        l_moved = 'l1' if self.player == 1 else 'l2'
        new[l_moved] = index
        new[l_moved + '_rot'] = rot
        new['player'] = self.next_player
        field = self.__class__(**new)
        field.assert_valid()
        return field

    def normalized(self, steps=False):
        # Bring L-piece 1 into rotation 0
        if self.player == 1:
            rot = -(self.l1_rot % 4)
            s = (('rotated', rot),)
            game = self.rotated(rot)
        else:
            rot = -(self.l2_rot % 4)
            s = ('swapped', ('rotated', rot))
            game = self.players_swapped().rotated(rot)

        if game.l1_rot >= 4:
            s = s + ('flipped',)
            game = game.flipped_horizontally()

        assert game.l1_rot == 0
        if steps:
            return game, s
        else:
            return game

    def players_swapped(self):
        new = self._as_dict()
        new['l1'] = self.l2_idx
        new['l1_rot'] = self.l2_rot
        new['l2'] = self.l1_idx
        new['l2_rot'] = self.l1_rot
        new['player'] = self.next_player
        return self.__class__(**new)

    @classmethod
    def action_swapped(cls, action):
        return action[:2] + (min(action[2:]), max(action[2:]))

    def assert_valid(self):
        assert self.n1_idx != self.n2_idx, "N-pieces collide"
        assert 0 <= self.n1_idx <= 15, "N-piece 1 is out of bound"
        assert 0 <= self.n2_idx <= 15, "N-piece 2 is out of bound"

        board = [[False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False]]
        for n in (self.n1, self.n2):
            x, y = _pos_to_image(*n)
            board[y][x] = True

        try:
            for pos, rot in ((self.l1, self.l1_rot), (self.l2, self.l2_rot)):
                x, y = _pos_to_image(*pos)
                for i, row in enumerate(self.ROTATIONS[rot]):
                    for j, ch in enumerate(row):
                        if ch != ' ':
                            assert not board[y + i][x + j], "L-pieces collide"
                            board[y + i][x + j] = True
        except IndexError:
            raise AssertionError("L-piece out of bound")

    @property
    def l1(self):
        return _index_to_pos(self[0])

    @property
    def l1_idx(self):
        return self[0]

    @property
    def l1_rot(self):
        return self[1]

    @property
    def l2(self):
        return _index_to_pos(self[2])

    @property
    def l2_idx(self):
        return self[2]

    @property
    def l2_rot(self):
        return self[3]

    @property
    def n1(self):
        return _index_to_pos(self[4])

    @property
    def n1_idx(self):
        return self[4]

    @property
    def n2(self):
        return _index_to_pos(self[5])

    @property
    def n2_idx(self):
        return self[5]

    @property
    def player(self):
        return self[6]

    @property
    def next_player(self):
        return 3 - self.player

    def rotated(self, times=1):
        """Rotate the field by 90deg in mathematically positive direction."""
        times = times % 4
        if times == 0:
            return self

        # sin = (0, 1, 0, -1)[times]
        # cos = (1, 0, -1, 0)[times]
        if times == 1:
            def rotate_point(pos): return (-pos[1], pos[0])
        elif times == 2:
            def rotate_point(pos): return (-pos[0], -pos[1])
        elif times == 3:
            def rotate_point(pos): return (pos[1], -pos[0])

        new = self._as_dict()
        new['n1'] = rotate_point(self.n1)
        new['n2'] = rotate_point(self.n2)

        for l in ('l1', 'l2'):
            rot = getattr(self, l + '_rot')
            rot_4 = rot % 4
            new[l + '_rot'] = (4 if rot >= 4 else 0) + (rot_4 + times) % 4

            x, y = rotate_point(_index_to_pos(getattr(self, l + '_idx')))
            rot_2 = rot % 2
            h, w = (2, 1)[rot_2], (1, 2)[rot_2]
            x += (0, 0, -w, -h)[times] * 2
            y += (0, w,  h,  0)[times] * 2
            new[l] = (x, y)

        return self.__class__(**new)

    @classmethod
    def action_rotated(cls, action, times=1):
        index, rot, n1_idx, n2_idx = action
        f = cls(**{
            'n1': n1_idx,
            'n2': n2_idx,
            'l1': index,
            'l1_rot': rot}).rotated(times)
        return f.l1_idx, f.l1_rot, f.n1_idx, f.n2_idx

    def flipped_horizontally(self):
        new = self._as_dict()
        new['l1'] = self.l1_idx + 1 - (self.l1_rot % 2) + 1 - 2 * (self.l1_idx % 4)
        new['l1_rot'] = (4, 7, 6, 5, 0, 3, 2, 1)[self.l1_rot]

        new['l2'] = self.l2_idx + 1 - (self.l2_rot % 2) + 1 - 2 * (self.l2_idx % 4)
        new['l2_rot'] = (4, 7, 6, 5, 0, 3, 2, 1)[self.l2_rot]

        x, y = self.n1
        new['n1'] = (-x, y)

        x, y = self.n2
        new['n2'] = (-x, y)
        return self.__class__(**new)

    @classmethod
    def action_flipped_horizontally(cls, action):
        index, rot, n1_idx, n2_idx = action
        f = cls(**{
            'n1': n1_idx,
            'n2': n2_idx,
            'l1': index,
            'l1_rot': rot}).flipped_horizontally()
        return f.l1_idx, f.l1_rot, f.n1_idx, f.n2_idx

    def as_ascii(self):
        lines = [[" "] + ["▁"] * 12 + [" "]] + \
                [["┃"] + [" "] * 12 + ["┃"] for _ in range(8)] + \
                [[" "] + ["▔"] * 12 + [" "]]

        for x, y in (self.n1, self.n2):
            _draw_ascii(lines, x, y, self.N)

        for idx, rot, rep in ((self.l1_idx, self.l1_rot, None), (self.l2_idx, self.l2_rot, '╳')):
            pattern = self.ROTATIONS[rot]
            if rep is not None:
                pattern = [line.replace('▓', rep) for line in pattern]
            x, y = _index_to_pos(idx)
            _draw_ascii(lines, x, y, pattern, scale=True)

        return "\n".join("".join(l) for l in lines)

    def as_rust(self):
        return f"LGame {{ l1: {self.l1_idx}, l1_rot: {self.l1_rot}, l2: {self.l2_idx}, l2_rot: {self.l2_rot}, n1: {self.n1_idx}, n2: {self.n2_idx}, player: {self.player} }}"

    def _as_dict(self):
        return {
            'l1': self.l1_idx, 'l1_rot': self.l1_rot,
            'l2': self.l2_idx, 'l2_rot': self.l2_rot,
            'n1': self.n1_idx, 'n2': self.n2_idx,
            'player': self.player}

    def __str__(self):
        return self.as_ascii()

    def __repr__(self):
        UC = ('▜', '▛▀', '▙', '▄▟', '▛', '▙▄', '▟', '▀▜')
        return 'LGame(L1: {} {}, L2: {} {}, N1: {}, N2: {})'.format(
            UC[self.l1_rot], _pos_to_image(*self.l1)[::-1],
            UC[self.l2_rot], _pos_to_image(*self.l2)[::-1],
            _pos_to_image(*self.n1)[::-1], _pos_to_image(*self.n2)[::-1])

    def _repr_svg_(self):
        x, y = _pos_to_image(*self.n1)
        n1_cx = 64.375 + (118.75 + 5) * x
        n1_cy = 64.375 + (118.75 + 5) * y

        x, y = _pos_to_image(*self.n2)
        n2_cx = 64.375 + (118.75 + 5) * x
        n2_cy = 64.375 + (118.75 + 5) * y

        svg = r"""
        <svg width="250px" height="280px" viewBox="0 0 500 560" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
            <g stroke="#000000" stroke-width="5" fill="none">
                <polygon points="2.5 2.5 497.5 2.5 497.5 497.5 2.5 497.5"></polygon>
                <path d="M126.25,499.5 L126.25,0"></path>
                <path d="M250,499.5 L250,0"></path>
                <path d="M373.75,499.5 L373.75,0"></path>
                <path d="M250.25,376 L250.25,-123.5" transform="translate(250.250000, 126.250000) rotate(-270.000000) translate(-250.250000, -126.250000) "></path>
                <path d="M250.25,499.75 L250.25,0.25" transform="translate(250.250000, 250.000000) rotate(-270.000000) translate(-250.250000, -250.000000) "></path>
                <path d="M250.25,623.5 L250.25,124" transform="translate(250.250000, 373.750000) rotate(-270.000000) translate(-250.250000, -373.750000) "></path>
            </g>
            <g fill="#000000" stroke="none">
                <circle cx="{}" cy="{}" r="59.375"></circle>
                <circle cx="{}" cy="{}" r="59.375"></circle>
            </g>
            <g stroke="none">
                <polyline fill="#D0021B" points="0 0 242.5 0 242.5 366.25 123.75 366.25 123.75 118.75 0 118.75" transform="{}"></polyline>
                <polyline fill="#4A90E2" points="0 0 242.5 0 242.5 366.25 123.75 366.25 123.75 118.75 0 118.75" transform="{}"></polyline>
            </g>
            <text id="Player-to-move:" font-family="ArialMT, Arial" font-size="18" font-weight="normal" letter-spacing="-0.472499967">
                <tspan x="11.0367182" y="538">Player to move:</tspan>
            </text>
            <rect width="24" height="24" fill="{}" y="519" x="144"/>
        </svg>""".format(
            n1_cx, n1_cy, n2_cx, n2_cy,
            _svg_transform(self.l1_idx, self.l1_rot),
            _svg_transform(self.l2_idx, self.l2_rot),
            ("#D0021B" if self.player == 1 else "#4A90E2"))
        return svg


def _svg_transform(index, rot):
    MAT = (
        (1, 0, 0, 1, 0, 0),
        (0, -1, 1, 0, 0, 242.5),
        (-1, 0, 0, -1, 242.5, 366.25),
        (0, 1, -1, 0, 366.25, 0),
        (-1, 0, 0, 1, 242.5, 0.),
        (0, 1, 1, 0, 0, 0.),
        (1, 0, 0, -1, 0, 366.25),
        (0, -1, -1, 0, 366.25, 242.5),
    )
    x, y = _pos_to_image(*_index_to_pos(index))
    mat = list(MAT[rot])
    mat[-2] += x * 123.75 + 5
    mat[-1] += y * 123.75 + 5
    return "matrix(" + ",".join(str(a) for a in mat) + ")"


class LGamePrecomputed(LGame):
    def valid_actions(self):
        return self.actions[self]

    def normalized(self, steps=False):
        if not steps:
            return self._normalized[self]
        else:
            return super().normalized(steps=True)


def _draw_ascii(lines, x, y, pattern, scale=False):
    x, y = _pos_to_image(x, y)
    x *= 3
    y *= 2
    if scale:
        pattern = [''.join([c * 3 for c in line]) for line in pattern]
        pattern = [x for line in pattern for x in [line] * 2]
    for y, line in enumerate(pattern, y + 1):
        dst = lines[y][x+1:x+1+len(line)]
        lines[y][x+1:x+1+len(line)] = [(a if a != ' ' else b) for a, b in zip(line, dst)]


def _mark_l_piece(board, x, y, rot, image=False):
    if not image:
        x, y = _pos_to_image(x, y)
    for i, row in enumerate(LGame.ROTATIONS[rot]):
        for j, ch in enumerate(row):
            if ch != ' ':
                board[y + i][x + j] = True


def _get_n_piece_moves(board, n1, n2):
    yield (n1, n2)
    for i, state in enumerate(x for line in board for x in line):
        if not state and i != n1 and i != n2:
            yield (n1, i)
            yield (i, n2)


def _get_l_piece_moves(board):
    # Look for vertical ones
    for x in range(4):
        for y in range(2):
            if board[y][x]:
                continue
            if not (board[y][x] == board[y + 1][x] == board[y + 2][x]):
                continue
            # We have an empty column
            if x <= 2:
                if not board[y][x + 1]:
                    yield x, y, 4
                if not board[y + 2][x + 1]:
                    yield x, y, 2
            if x >= 1:
                if not board[y][x - 1]:
                    yield x - 1, y, 0
                if not board[y + 2][x - 1]:
                    yield x - 1, y, 6

    # Look for horizontal ones
    for y in range(4):
        for x in range(2):
            if board[y][x]:
                continue
            if not (board[y][x] == board[y][x + 1] == board[y][x + 2]):
                continue
            # We have an empty row
            if y <= 2:
                if not board[y + 1][x]:
                    yield x, y, 1
                if not board[y + 1][x + 2]:
                    yield x, y, 7
            if y >= 1:
                if not board[y - 1][x]:
                    yield x, y - 1, 5
                if not board[y - 1][x + 2]:
                    yield x, y - 1, 3


def _index_to_pos(idx):
    return 2 * (idx % 4) - 3, -2 * (idx // 4) + 3


def _pos_to_index(pos):
    (x, y) = tuple(pos)
    return (3 + x) // 2 + 4 * (3 - y) // 2


def _pos_to_image(x, y):
    """Convert centered coordinates to top-left coordinates."""
    return (3 + x) // 2, (3 - y) // 2


def _image_to_pos(x, y):
    return 2 * x - 3, 3 - 2 * y


def _image_to_index(x, y):
    return x + 4 * y


def _precompute():
    from .statesearch import find_all_states
    from joblib import Parallel, delayed
    print("Compute all states...", end='', flush=True)
    states, terminals = find_all_states(LGame)
    print("OK, n={} ({} terminal)".format(len(states), len(terminals)))

    print("Compute all actions...", end='', flush=True)
    actions = dict(Parallel(n_jobs=-1)(
        delayed(lambda s: (s, list(s.valid_actions())))(s)
        for s in states))
    print("OK.")

    print("Compute unique states", end='', flush=True)
    unique_states = set(s.normalized() for s in states)
    unique_terminals = set(s.normalized() for s in terminals)
    print("OK.")

    print("Compute normalized states", end='', flush=True)
    normalized = dict((s, s.normalized()) for s in states)
    return {
        'states': [tuple(s) for s in states],
        'terminals': [tuple(s) for s in terminals],
        'unique_states': [tuple(s) for s in unique_states],
        'unique_terminals': [tuple(s) for s in unique_terminals],
        '_normalized': dict((tuple(s), tuple(ns)) for s, ns in normalized.items()),
        'actions': dict((tuple(s), actions)
                        for s, actions in actions.items())}


with GzipFile(path.join(path.dirname(__file__), 'lgame.gz'), 'rb') as f:
    for attr, val in pickle.load(f).items():
        if isinstance(val, list):
            val = [LGamePrecomputed(s) for s in val]
        elif isinstance(val, dict):
            val = {LGamePrecomputed(s): (LGamePrecomputed(x) if isinstance(x, tuple) else x) for s, x in val.items()}
        setattr(LGame, attr, val)
