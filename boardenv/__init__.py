from gym.envs.registration import register
from .env import *
from .reversi import *
from .kinarow import *
from .go import *
from .version import *
from .cchess import *
from .gobang import *

register(
    id='Reversi-v0',
    entry_point='boardenv:ReversiEnv',
)

register(
    id='KInARow-v0',
    entry_point='boardenv:KInARowEnv',
)

register(
    id='Gomuku-v0',
    entry_point='boardenv:KInARowEnv',
    kwargs={
        'board_shape': 15,
        'target_length': 5,
    }
)
register(
    id='Gobang-v0',
    entry_point='boardenv:GobangEnv',
    kwargs={
        'board_shape': 8,
        'target_length': 4,
    }
)

register(
    id='TicTacToe-v0',
    entry_point='boardenv:KInARowEnv',
    kwargs={
        'board_shape': 3,
        'target_length': 3,
    }
)

register(
    id='Go-v0',
    entry_point='boardenv:GoEnv',
)

register(
    id='ChineseChess-v0',
    entry_point='boardenv:CChessEnv'
)
