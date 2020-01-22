import copy
import itertools

import numpy as np

from .env import *


class GobangEnv(BoardGameEnv):

    def __init__(self, board_shape=15, target_length=5,
                 illegal_action_mode='pass', render_characters='+ox'):
        super().__init__(board_shape=board_shape,
                         illegal_action_mode=illegal_action_mode,
                         render_characters=render_characters)
        self.target_length = target_length

    def get_winner(self, state):
        board, _ = state
        for player in [BLACK, WHITE]:
            for x in range(board.shape[0]):
                for y in range(board.shape[1]):
                    for dx, dy in [(1, -1), (1, 0), (1, 1), (0, 1)]:  # loop on the 8 directions
                        xx, yy = x, y
                        for count in itertools.count():
                            if not is_index(board, (xx, yy)) or board[xx, yy] != player:
                                break
                            xx, yy = xx + dx, yy + dy
                            if count >= self.target_length:
                                return player
        for player in [BLACK, WHITE]:
            if self.has_valid((board, player)):
                return None
        return 0

    def render(self, mode='human'):
        render_characters = '+ox'
        rd = {EMPTY: '-', BLACK: 'o', WHITE: 'x'}
        for t in self.board:
            print(''.join([rd[pc] + ' ' for pc in t]))
