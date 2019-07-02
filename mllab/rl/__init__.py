import math
from itertools import count

from .lgame import LGamePrecomputed as LGame
from .tictactoe import TicTacToe

from IPython.display import display
import time


def self_play(initial_state, policy, policy2=None, *, sleep=2, max_steps=None):
    s = initial_state() if isinstance(initial_state, type) else initial_state
    if sleep > 0:
        handle = display(s, display_id=True)
    steps = count()
    if policy2 is None:
        policy2 = policy
    while s.valid_actions():
        action = policy[s]
        policy, policy2 = policy2, policy
        s = s.apply_action(action)
        if next(steps) == max_steps:
            pl = "moves" if max_steps != 1 else "move"
            print("Stopped after {} {}.".format(max_steps, pl))
            break
        if sleep > 0:
            time.sleep(sleep)
            handle.update(s)
    else:
        if sleep > 0:
            print("Game finished.")
    return s, next(steps)


class Policy:
    def __init__(self, game, for_unique_states):
        self._for_unique_states = for_unique_states
        self._game = game

    def __getitem__(self, s):
        a = self._for_unique_states[s.normalized()]
        return self._game.denormalize_action(a, s)
