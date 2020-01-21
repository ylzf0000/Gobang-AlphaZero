from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.Qt import QPixmap, QPainter

from agent import *
from boardenv.cchess import *


class QChessBoard(QWidget):
    square_size = 72
    board_edge = 8
    board_width = board_edge + square_size * 9 + board_edge
    board_height = board_edge + square_size * 10 + board_edge
    pc_str = ('K', 'A', 'B', 'N', 'R', 'C', 'P', '',
              'RK', 'RA', 'RB', 'RN', 'RR', 'RC', 'RP', '',
              'BK', 'BA', 'BB', 'BN', 'BR', 'BC', 'BP', '')
    selected_str = ('', 'S')
    pc_path = 'images\\IMAGES_X\\COMIC\\'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board_jpg = QtGui.QPixmap('images\\IMAGES_X\\WOOD.JPG')
        self.oos_gif = QtGui.QPixmap('images\\IMAGES_X\\COMIC\\OOS.GIF')
        self.painter = QPainter()
        self.agent = None
        self.is_flipped = False
        # self.game_state = None
        self.sq_selected = (-1, -1)
        self.mv_cur = (-1, -1, -1, -1)
        self.mv_last = (-1, -1, -1, -1)

    def set_agent(self, agent: AlphaZeroAgent):
        self.agent = agent

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        super().mousePressEvent(e)
        x, y = e.x(), e.y()
        x = round((x - 40) / self.square_size)
        y = round((y - 40) / self.square_size)
        agent = self.agent
        env = agent.env
        player = env.player
        pc = env.board[y][x]
        self_side = side_tag(player)
        if is_self_piece_by_tag(pc, self_side):
            self.sq_selected = (x, y)
            self.repaint()
            return

        if self.sq_selected != (-1, -1) and not is_self_piece_by_tag(pc, self_side):
            if not is_legalmove(env.board,
                                self.sq_selected[0], self.sq_selected[1], x, y,
                                player):
                self.sq_selected = (-1, -1)
                return
            """
            玩家走子
            """
            action = (dict_mv[mv_to_str(self.sq_selected[0], self.sq_selected[1], x, y)],)
            env.step(action)
            self.mv_last = self.mv_cur
            self.mv_cur = (self.sq_selected[0], self.sq_selected[1], x, y)
            self.sq_selected = (-1, -1)
            self.repaint()
            state = (env.board, env.player, env.depth)
            game_state = env.get_winner(state)
            if game_state != None:
                QMessageBox.information(self, '游戏结束！', '游戏结束！')
                return
            """
            AI走子
            """
            self.mv_last = self.mv_cur
            state = (env.board, env.player, env.depth)
            action = agent.decide(state)
            env.step(action)
            self.mv_cur = str_to_mv(labels_mv[action[0]])
            self.sq_selected = (-1, -1)
            self.repaint()
            state = (env.board, env.player, env.depth)
            game_state = env.get_winner(state)
            print(game_state)
            if game_state != None:
                QMessageBox.information(self, '游戏结束！', '游戏结束！')
                return

    def printBoard(self):
        self.env.render()

    def drawOOS(self, x, y):
        if x == -1 or y == -1:
            return
        x = self.board_edge + self.square_size * x
        y = self.board_edge + self.square_size * y
        self.painter.drawPixmap(x, y, self.oos_gif)

    def paintEvent(self, e):
        super().paintEvent(e)
        env = self.agent.env
        board = env.board
        self.painter.begin(self)
        self.painter.drawPixmap(0, 0, self.board_jpg)
        for j in range(env.BOARD_SHAPE[0]):
            for i in range(env.BOARD_SHAPE[1]):
                pc = board[j][i]
                if pc == 0:
                    continue
                gif_path = f'{self.pc_path}{self.pc_str[pc]}' \
                           f'{self.selected_str[int(self.sq_selected == (i, j))]}.GIF'
                gif = QtGui.QPixmap(gif_path)
                x = i
                y = j
                px = self.board_edge + self.square_size * x
                py = self.board_edge + self.square_size * y
                self.painter.drawPixmap(px, py, gif)
        self.drawOOS(self.mv_last[0], self.mv_last[1])
        self.drawOOS(self.mv_last[2], self.mv_last[3])
        self.drawOOS(self.mv_cur[0], self.mv_cur[1])
        self.drawOOS(self.mv_cur[2], self.mv_cur[3])
        self.painter.end()
