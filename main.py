# coding=utf-8
import argparse
from agent import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
if __name__ == '__main__':
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
