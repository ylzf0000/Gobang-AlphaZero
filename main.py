# coding=utf-8
import argparse
from agent import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='训练AI或与AI对弈')
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--train', action='store_true', help='开始训练')
    group1.add_argument('--play', action='store_true', help='与AI对局')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--cmd', action='store_true', help='命令行')
    group2.add_argument('--window', action='store_true', help='图形化')
    args = parser.parse_args()
    if not args.train and not args.play:
        args.train = True
    if not args.cmd and not args.window:
        args.cmd = True
    # 运行参数
    if args.train:
        train(args.cmd)
    elif args.play:
        play()


if __name__ == '__main__':
    # from boardenv import env
    #
    # board = np.array([i for i in range(1, 82)]).reshape((9, 9))
    # print(board)
    # loc = (3, 7)
    # print(board[loc])
    # boards = extend_board(board)
    # locs = extend_location(loc, board.shape)
    # for i in range(len(boards)):
    #     board = boards[i]
    #     loc = tuple(locs[i])
    #     print(board[loc])
    # for i in range(len(boards)):
    #     board = boards[i]
    #     for y in range(board.shape[0]):
    #         ok = 0
    #         for x in range(board.shape[1]):
    #             pc = board[y, x]
    #             if pc == 35:
    #                 print(y, x)
    #                 ok = 1
    #                 break
    #         if ok == 1:
    #             break
    # print(boards)
    # print(locs)
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    main()
