import os
import math
import time
import logging

from tensorflow import keras
import numpy as np
import pandas as pd
import gym
import tensorflow as tf

import boardenv
from boardenv import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')
env = gym.make('Gobang-v0')


def measure_time():
    def wraps(func):
        def mesure(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            # logger.info("function %s use time %s"%(func.__name__,(end-start)))
            print("function %s use time %s" % (func.__name__, (end - start)))
            return res

        return mesure

    return wraps


def residual(x, filters, kernel_sizes=3, strides=1, activations='relu',
             regularizer=keras.regularizers.l2(1e-4)):
    shortcut = x
    for i, filte in enumerate(filters):
        kernel_size = kernel_sizes if isinstance(kernel_sizes, int) \
            else kernel_sizes[i]
        stride = strides if isinstance(strides, int) else strides[i]
        activation = activations if isinstance(activations, str) \
            else activations[i]
        z = keras.layers.Conv2D(filte, kernel_size, strides=stride,
                                padding='same', kernel_regularizer=regularizer,
                                bias_regularizer=regularizer)(x)
        y = keras.layers.BatchNormalization()(z)
        if i == len(filters) - 1:
            y = keras.layers.Add()([shortcut, y])
        x = keras.layers.Activation(activation)(y)
    return x


class AlphaZeroAgent:
    def __init__(self, env, net_scale, batches=1, batch_size=4096,
                 kwargs=None, load=None, sim_count=800,
                 c_init=1.25, c_base=19652., prior_exploration_fraction=0.25):

        if kwargs is None:
            kwargs = {}
        self.env = env
        self.board = np.zeros_like(env.board)
        self.batches = batches
        self.batch_size = batch_size
        self.net_scale = net_scale
        if self.net_scale == 'big':
            self.model_filename = './gobang_model_big.h5'
        elif self.net_scale == 'small':
            self.model_filename = './gobang_model_small.h5'
        if os.path.isfile(self.model_filename):
            self.net = keras.models.load_model(self.model_filename, custom_objects={
                'categorical_crossentropy_2d': self.categorical_crossentropy_2d})
        else:
            self.net = self.build_network(**kwargs)
        self.reset_mcts()
        self.sim_count = sim_count  # MCTS 次数
        self.c_init = c_init  # PUCT 系数
        self.c_base = c_base  # PUCT 系数
        self.prior_exploration_fraction = prior_exploration_fraction

    def categorical_crossentropy_2d(self, y_true, y_pred):
        labels = tf.reshape(y_true, [-1, self.board.size])
        preds = tf.reshape(y_pred, [-1, self.board.size])
        return keras.losses.categorical_crossentropy(labels, preds)

    def build_network(self, conv_filters, residual_filters, policy_filters,
                      learning_rate=0.001, regularizer=keras.regularizers.l2(1e-4)):
        # 公共部分
        inputs = keras.Input(shape=self.board.shape)
        x = keras.layers.Reshape(self.board.shape + (1,))(inputs)
        for conv_filter in conv_filters:
            z = keras.layers.Conv2D(conv_filter, 3, padding='same',
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(x)
            y = keras.layers.BatchNormalization()(z)
            x = keras.layers.ReLU()(y)
        for residual_filter in residual_filters:
            x = residual(x, filters=residual_filter, regularizer=regularizer)
        intermediates = x

        # 概率部分
        for policy_filter in policy_filters:
            z = keras.layers.Conv2D(policy_filter, 3, padding='same',
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(x)
            y = keras.layers.BatchNormalization()(z)
            x = keras.layers.ReLU()(y)
        logits = keras.layers.Conv2D(1, 3, padding='same',
                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        flattens = keras.layers.Flatten()(logits)
        softmaxs = keras.layers.Softmax()(flattens)
        probs = keras.layers.Reshape(self.board.shape)(softmaxs)

        # 价值部分
        z = keras.layers.Conv2D(1, 3, padding='same',
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer)(intermediates)
        y = keras.layers.BatchNormalization()(z)
        x = keras.layers.ReLU()(y)
        flattens = keras.layers.Flatten()(x)
        vs = keras.layers.Dense(1, activation=keras.activations.tanh,
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer)(flattens)

        model = keras.Model(inputs=inputs, outputs=[probs, vs])

        loss = [self.categorical_crossentropy_2d, keras.losses.MSE]
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def reset_mcts(self):
        def zero_board_factory():  # 用于构造 default_dict
            return np.zeros_like(self.board, dtype=float)

        self.q = collections.defaultdict(zero_board_factory)
        # q值估计: board -> board
        self.count = collections.defaultdict(zero_board_factory)
        # q值计数: board -> board
        self.policy = {}  # 策略: board -> board
        self.valid = {}  # 有效位置: board -> board
        self.winner = {}  # 赢家: board -> None or int

    def decide(self, observation, greedy=False, return_prob=False):
        # 计算策略
        board, player = observation
        canonical_board = player * board
        s = boardenv.strfboard(canonical_board)
        while self.count[s].sum() < self.sim_count:  # 多次 MCTS 搜索
            self.search(canonical_board, prior_noise=True)
        prob = self.count[s] / self.count[s].sum()

        # 采样
        location_index = np.random.choice(prob.size, p=prob.reshape(-1))
        location = np.unravel_index(location_index, prob.shape)
        if return_prob:
            return location, prob
        return location

    def learn(self, dfs):
        df = pd.concat(dfs).reset_index(drop=True)
        for batch in range(self.batches):
            indices = np.random.choice(len(df), size=self.batch_size)
            players, boards, probs, winners = (np.stack(
                df.loc[indices, field]) for field in df.columns)
            canonical_boards = players[:, np.newaxis, np.newaxis] * boards
            vs = (players * winners)[:, np.newaxis]
            self.net.fit(canonical_boards, [probs, vs], verbose=0)  # 训练
        self.reset_mcts()

    def search(self, board, prior_noise=False):  # MCTS 搜索
        s = boardenv.strfboard(board)

        if s not in self.winner:
            self.winner[s] = self.env.get_winner((board, BLACK))  # 计算赢家
        if self.winner[s] is not None:  # 赢家确定的情况
            return self.winner[s]

        if s not in self.policy:  # 未计算过策略的叶子节点
            pis, vs = self.net.predict(board[np.newaxis])
            pi, v = pis[0], vs[0]
            valid = self.env.get_valid((board, BLACK))
            masked_pi = pi * valid
            total_masked_pi = np.sum(masked_pi)
            if total_masked_pi <= 0:  # 所有的有效动作都没有概率，偶尔可能发生
                masked_pi = valid  # workaround
                total_masked_pi = np.sum(masked_pi)
            self.policy[s] = masked_pi / total_masked_pi
            self.valid[s] = valid
            return v

        # PUCT 上界计算
        count_sum = self.count[s].sum()
        coef = (self.c_init + np.log1p((1 + count_sum) / self.c_base)) * \
               math.sqrt(count_sum) / (1. + self.count[s])
        if prior_noise:  # 先验噪声
            alpha = 1. / self.valid[s].sum()
            noise = np.random.gamma(alpha, 1., board.shape)
            noise *= self.valid[s]
            noise /= noise.sum()
            prior = (1. - self.prior_exploration_fraction) * \
                    self.policy[s] + \
                    self.prior_exploration_fraction * noise
        else:
            prior = self.policy[s]
        ub = np.where(self.valid[s], self.q[s] + coef * prior, np.nan)
        location_index = np.nanargmax(ub)
        location = np.unravel_index(location_index, board.shape)

        (next_board, next_player), _, _, _ = self.env.next_step(
            (board, BLACK), np.array(location))
        next_canonical_board = next_player * next_board
        next_v = self.search(next_canonical_board)  # 递归搜索
        v = next_player * next_v

        self.count[s][location] += 1
        self.q[s][location] += (v - self.q[s][location]) / \
                               self.count[s][location]
        return v


def flip_trajectory(df_trajectory: pd.DataFrame):
    raw_trajectory = list(df_trajectory.values.tolist())
    winner = raw_trajectory[0][3]
    trajectories = [[] for _ in range(8)]
    dfs_trajectory = []
    for row in raw_trajectory:
        player = row[0]
        board = row[1]
        prob = row[2]
        # winner = row[3]
        boards = extend_board(board)
        probs = extend_board(prob)
        for i in range(len(boards)):
            trajectories[i].append((player, boards[i], probs[i]))
    for trajectory in trajectories:
        df = pd.DataFrame(trajectory,
                          columns=['player', 'board', 'prob'])
        df['winner'] = winner
        dfs_trajectory.append(df)
    return dfs_trajectory


@measure_time()
def self_play(env, agent, iteration, episode, return_trajectory=False, verbose=False):
    trajectory = [] if return_trajectory else None
    observation = env.reset()
    winner = None
    for step in itertools.count():
        board, player = observation
        action, prob = agent.decide(observation, return_prob=True)
        if verbose:
            env.render()
            logging.info(f'训练{iteration}，回合{episode}，第{step}步，玩家{player2str(player)}，动作{action}')
        observation, winner, done, _ = env.step(action)
        if return_trajectory:
            trajectory.append((player, board, prob))
        if done:
            if verbose:
                env.render()
                logging.info(f'训练{iteration}，回合{episode}，对弈了{step}步, 赢家为{player2str(winner)}')
            break
    if return_trajectory:
        df_trajectory = pd.DataFrame(trajectory,
                                     columns=['player', 'board', 'prob'])
        df_trajectory['winner'] = winner
        return df_trajectory
    else:
        return winner


def train_args(scale):
    train_iterations = 0
    train_episodes_per_iteration = 0
    batches = 0
    batch_size = 0
    if scale == 'big':
        """
        AlphaZero 参数，可用来求解比较大型的问题（如五子棋）
        """
        train_iterations = 700000  # 训练迭代次数
        train_episodes_per_iteration = 5000  # 每次迭代自我对弈回合数
        batches = 10  # 每回合进行几次批学习
        batch_size = 4096  # 批学习的批大小
    elif scale == 'small':
        """
        小规模参数，用来初步求解比较小的问题（如井字棋）
        """
        train_iterations = 100
        train_episodes_per_iteration = 100
        batches = 2
        batch_size = 64
    return train_iterations, train_episodes_per_iteration, batches, batch_size


def net_args(scale):
    sim_count = 0
    net_kwargs = {}
    net_scale = ''
    if scale == 'big':
        """
        AlphaZero 参数，可用来求解比较大型的问题（如五子棋）
        """
        sim_count = 800  # MCTS需要的计数
        net_kwargs = {}
        net_kwargs['conv_filters'] = [256, ]
        net_kwargs['residual_filters'] = [[256, 256], ] * 19
        net_kwargs['policy_filters'] = [256, ]
        net_scale = 'big'
    elif scale == 'small':
        """
        小规模参数，用来初步求解比较小的问题（如井字棋）
        """
        sim_count = 200
        net_kwargs = {}
        net_kwargs['conv_filters'] = [256, ]
        net_kwargs['residual_filters'] = [[256, 256], ] * 7
        net_kwargs['policy_filters'] = [256, ]
        net_scale = 'small'
    return sim_count, net_kwargs, net_scale


def train(cmd, scale='small'):
    print(sys._getframe().f_code.co_name)
    train_iterations, train_episodes_per_iteration, \
    batches, batch_size = train_args(scale)
    sim_count, net_kwargs, net_scale = net_args(scale)
    agent = AlphaZeroAgent(env=env, net_scale=net_scale,
                           kwargs=net_kwargs, sim_count=sim_count,
                           batches=batches, batch_size=batch_size)
    for iteration in range(train_iterations):
        # 自我对弈
        dfs_trajectory = []
        for episode in range(train_episodes_per_iteration):
            logging.info(f'训练 {iteration} 回合 {episode}开始')
            df_trajectory = self_play(env, agent, iteration, episode,
                                      return_trajectory=True, verbose=True)

            logging.info(f'训练 {iteration} 回合 {episode}: 收集到 {len(df_trajectory)} 条经验')
            dfs_trajectory += flip_trajectory(df_trajectory)
            # dfs_trajectory.append(df_trajectory)

        # 利用经验进行学习
        agent.learn(dfs_trajectory)
        keras.models.save_model(agent.net, agent.model_filename)
        logging.info('训练 {}: 学习完成'.format(iteration))

        # 演示训练结果
        # self_play(env, agent, iteration, episode, verbose=True)


def play(scale='small'):
    print(sys._getframe().f_code.co_name)
    train_iterations, train_episodes_per_iteration, \
    batches, batch_size = train_args(scale)
    sim_count, net_kwargs, net_scale = net_args(scale)
    agent = AlphaZeroAgent(env=env, net_scale=net_scale,
                           kwargs=net_kwargs, sim_count=sim_count,
                           batches=batches, batch_size=batch_size)
    from mainwindow import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.widgetBoard.set_agent(agent)
    mainWindow.show()
    sys.exit(app.exec_())
