from gym.envs.registration import register
from .version import *
from .gobang import *

register(
    id='Gobang-v0',
    entry_point='boardenv:GobangEnv',
    kwargs={
        'board_shape': 6,
        'target_length': 4,
    }
)
