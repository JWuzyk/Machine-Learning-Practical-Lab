import pickle
from os import path
from gzip import GzipFile
from .game import TwoPlayerFiniteGame


class TicTacToe(TwoPlayerFiniteGame):
    X = 1
    O = 2
    EMPTY = 0

    def __new__(cls, state=None):
        if state is None:
            return tuple.__new__(cls, (
                (cls.EMPTY, cls.EMPTY, cls.EMPTY),
                (cls.EMPTY, cls.EMPTY, cls.EMPTY),
                (cls.EMPTY, cls.EMPTY, cls.EMPTY), cls.X))
        else:
            return tuple.__new__(cls, state)

    def normalized(self, steps=False):
        """Return a new normalized board.

        This is a unique element of the equivalence class of all rotations and
        reflections."""
        if steps:
            key = lambda f: f[0]._key()
        else:
            key = lambda f: f._key()
        return max(self._symmetries(steps), key=key )

    def valid_actions(self):
        """Get the valid actions for a given player."""
        if not hasattr(self, '_valid_actions'):
            self._valid_actions = []
            if not self.winner():
                for y, row in enumerate(self[:-1]):
                    for x, char in enumerate(row):
                        if char == self.EMPTY:
                            self._valid_actions.append((y, x))
        return self._valid_actions

    @property
    def player(self):
        return self[-1]

    def winner(self):
        """Check if there is a winner."""
        if not hasattr(self, '_winner'):
            options = (((0, 0), (1, 1), (2, 2)),  # diagonals
                       ((0, 2), (1, 1), (2, 0)),
                       ((0, 0), (0, 1), (0, 2)),  # rows
                       ((1, 0), (1, 1), (1, 2)),
                       ((2, 0), (2, 1), (2, 2)),
                       ((0, 0), (1, 0), (2, 0)),  # columns
                       ((0, 1), (1, 1), (2, 1)),
                       ((0, 2), (1, 2), (2, 2)))
            for a, b, c in options:
                player = self[a]
                if player == self.EMPTY:
                    continue
                if player == self[b] == self[c]:
                    self._winner = player
                    break
            else:
                self._winner = None
        return self._winner

    def apply_action(self, action):
        """Apply an action, i.e. a move as returned by `valid_actions()`."""
        i, j = action
        state = self[:i] + \
            (self[i][:j] + (self[-1],) + self[i][j+1:],) + \
            self[i+1:-1] + (self.X if self[-1] == self.O else self.O,)
        return self.__class__(state)

    def rotated(self, times=1):
        times = times % 4
        if times == 0:
            return self

        if times == 1:
            state = ((self[0][2], self[1][2], self[2][2]),
                     (self[0][1], self[1][1], self[2][1]),
                     (self[0][0], self[1][0], self[2][0]), self[-1])
        elif times == 2:
            state = ((self[2][2], self[2][1], self[2][0]),
                     (self[1][2], self[1][1], self[1][0]),
                     (self[0][2], self[0][1], self[0][0]), self[-1])
        else:
            state = ((self[2][0], self[1][0], self[0][0]),
                     (self[2][1], self[1][1], self[0][1]),
                     (self[2][2], self[1][2], self[0][2]), self[-1])

        return self.__class__(state)

    @classmethod
    def action_rotated(cls, action, times=1):
        y, x = action
        f = TicTacToe().apply_action(action).rotated(times)
        for y, row in enumerate(f[:-1]):
            if row == (cls.EMPTY, cls.EMPTY, cls.EMPTY):
                continue
            return y, row.index(cls.X)

    def flipped_horizontally(self, action=None):
        return self.__class__((self[0][::-1], self[1][::-1], self[2][::-1], self[-1]))

    @classmethod
    def action_flipped_horizontally(cls, action):
        y, x = action
        return y, 2 - x

    def as_ascii(self):
        output = [["┌───┬───┬───┐"],
                  ["│", "   ", "│", "   ", "│", "   ", "│"],
                  ["│", "   ", "│", "   ", "│", "   ", "│"],
                  ["├───┼───┼───┤"],
                  ["│", "   ", "│", "   ", "│", "   ", "│"],
                  ["│", "   ", "│", "   ", "│", "   ", "│"],
                  ["├───┼───┼───┤"],
                  ["│", "   ", "│", "   ", "│", "   ", "│"],
                  ["│", "   ", "│", "   ", "│", "   ", "│"],
                  ["└───┴───┴───┘"]]
        for y, row in enumerate(self[:-1]):
            for x, char in enumerate(row):
                i = 1 + y * 3
                j = 1 + x * 2
                if char == self.O:
                    output[i][j] = "▞▀▚"
                    output[i+1][j] = "▚▃▞"
                elif char == self.X:
                    output[i][j] = "▚▂▞"
                    output[i+1][j] = "▞▔▚"

        return "\n".join("".join(row) for row in output)

    def __str__(self):
        return self.as_ascii()

    def __repr__(self):
        CHAR_MAP = {self.EMPTY: '☐', self.X: 'x', self.O: 'o'}
        rows = ["|".join(CHAR_MAP[c] for c in row) for row in self[:3]]
        return "TicTacToe({}, {}, {})".format(*rows)

    def _repr_svg_(self):
        svg = r"""
        <svg width="250px" height="280px" viewBox="0 0 500 560" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
            <defs>
                <g id="cross" stroke="#000000" stroke-width="5" fill="none" stroke-linecap="square">
                    <path d="M0.390625,0.390625 L99.6124507,99.6124507" />
                    <path d="M99.609375,0.390625 L0.38754926,99.6124507" />
                </g>
                <g id="circle" stroke="#000000" stroke-width="5" fill="none" stroke-linecap="square">
                    <circle cx="45.666" cy="45.666" r="57.5"/>
                </g>
            </defs>
            <g stroke="#000000" stroke-width="5" fill="none" stroke-linecap="square">
                <path d="M0,163.33 L500,163.33" />
                <path d="M0,326.66 L500,326.66" />
                <path d="M163.33,0 L163.33,500" />
                <path d="M326.66,0 L326.66,500" />
            """
        for y, row in enumerate(self[:-1]):
            for x, char in enumerate(row):
                if char == self.EMPTY:
                    continue
                dx = (163.333 + 2.5) * x
                dy = (163.333 + 2.5) * y
                if char == self.O:
                    cx = dx + 163.333/2
                    cy = dy + 163.333/2
                    svg += """<circle cx="{}" cy="{}" r="57.5" />""".format(cx, cy)
                else:
                    dx += 31.666
                    dy += 31.666
                    svg += """<g transform="translate({}, {})"><use xlink:href="#cross" /></g>""".format(dx, dy)
        return svg + """</g>
            <text id="Player-to-move:" font-family="ArialMT, Arial" font-size="18" font-weight="normal" letter-spacing="-0.472499967">
                <tspan x="11.0367182" y="538">Player to move:</tspan>
            </text><g transform="translate(150, 521), scale(.25)"><use xlink:href="#""" + \
            ("cross" if self.player == self.X else "circle") + """"/></g></svg>"""

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self[key[0]][key[1]]
        else:
            return super().__getitem__(key)

    def _symmetries(self, steps):
        for i in range(4):
            rotated = self.rotated(i)
            yield (rotated, (('rotated', i),)) if steps else rotated
            flipped = rotated.flipped_horizontally()
            yield (flipped, (('rotated', i), 'flipped')) if steps else flipped

    def _key(self):
        if not hasattr(self, '_key_cached'):
            flatten = (x for row in self[:-1] for x in row)
            self._key_cached = sum(x << 2 * i for i, x in enumerate(flatten))
        return self._key_cached


if not hasattr(TicTacToe, 'UNIQUE_STATES'):
    with GzipFile(path.join(path.dirname(__file__), 'tictactoe.states.gz'), 'rb') as f:
        TicTacToe.unique_states = [TicTacToe(s) for s in pickle.load(f)]
