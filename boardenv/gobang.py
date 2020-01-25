import copy
import itertools

import numpy as np

from .env import *


def net_input(board, player, last_action):
    h, w = board.shape[0], board.shape[1]
    input = np.zeros(shape=(h, w, 4))
    t = {player: 0, -player: 1}
    for y in range(h):
        for x in range(w):
            pc = board[y, x]
            if pc != 0:
                input[y, x, t[pc]] = 1
    input[last_action[0], last_action[1], 2] = 1
    if player == BLACK:
        input[:, :, 3] = np.ones((h, w))
    return input


class GobangEnv(BoardGameEnv):

    def step(self, action):
        state = (self.board, self.player)
        next_state, reward, done, info = self.next_step(state, action)
        self.board, self.player = next_state
        self.last_action = action
        return next_state, reward, done, info

    def __init__(self, board_shape, target_length,
                 illegal_action_mode='pass', render_characters='-ox'):
        super().__init__(board_shape=board_shape,
                         illegal_action_mode=illegal_action_mode,
                         render_characters=render_characters)
        self.target_length = target_length
        self.last_action = None

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
                            if count + 1 >= self.target_length:
                                return player
        for player in [BLACK, WHITE]:
            if self.has_valid((board, player)):
                return None
        return 0

    def render(self, mode='human'):
        render_characters = {EMPTY: '-', BLACK: 'o', WHITE: 'x'}
        for t in self.board:
            print(''.join([render_characters[pc] + ' ' for pc in t]))
