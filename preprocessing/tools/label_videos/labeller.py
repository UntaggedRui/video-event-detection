#! /usr/bin/env python
# coding=utf-8
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import os
import cv2 as cv
import time

class MainWindow(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)
        self.resize(550, 360)
        self.setWindowTitle('shot-event-labeller')
        self.status = 0  # 0 is init status;1 is play video; 2 is capture video
        self.image = QImage()

        self.videoList = QListWidget()
        self.labelList = QListWidget()
        self.loadLabels()

        # 初始化按钮
        self.selectbtn = QPushButton('select')
        self.playbtn = QPushButton('play')
        exitbtn = QPushButton('exit')

        # 界面布局
        vbox = QVBoxLayout()
        vbox.addWidget(self.labelList)
        vbox.addWidget(self.selectbtn)
        vbox.addWidget(self.playbtn)
        vbox.addWidget(exitbtn)

        self.piclabel = QLabel('pic')
        self.piclabel.setFixedSize(640,360)
        self.piclabel.setScaledContents(True)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addStretch(1)
        hbox.addWidget(self.piclabel)
        hbox.addWidget(self.videoList)

        self.setLayout(hbox)

        # 加载初始页面
        #stat, frame = self.playcapture.read()
        #frame = cv.cv.fromarray(frame)
        #cv.cv.SaveImage('1.jpg', frame)

        #if self.image.load("1.jpg"):
        #    self.piclabel.setPixmap(QPixmap.fromImage(self.image))

        # 设定定时器
        self.timer = Timer()  # 录制视频
        self.playtimer = Timer("updatePlay()")  # 播放视频

        # 信号--槽
        self.connect(self.videoList, SIGNAL("itemDoubleClicked(QListWidgetItem*)"),
                     self.PlayItemVideo)
        self.connect(self.playtimer, SIGNAL("updatePlay()"),
                     self.PlayVideo)
        self.connect(self.selectbtn, SIGNAL("clicked()"),
                     self.SelectPath)
        self.connect(self.playbtn, SIGNAL("clicked()"),
                     self.VideoPlayPause)
        self.connect(exitbtn, SIGNAL("clicked()"),
                     app, SLOT("quit()"))

        if os.path.exists('./_temp') == False:
            os.mkdir('./_temp')

    def loadLabels(self):
        file = open('labels.txt')
        line = file.readline()
        while line:
            line = line.strip('\n')
            self.labelList.addItem(line)
            line = file.readline()
        file.close()




    def PlayVideo(self):
        stat, frame = self.playcapture.read()
        if stat == False:
            self.playtimer.stop()
        else:
            frame = cv.cv.fromarray(frame)
            cv.cv.SaveImage('./_temp/play_temp.jpg', frame)
            self.image.load("./_temp/play_temp.jpg")
            self.piclabel.setPixmap(QPixmap.fromImage(self.image))

    def PlayItemVideo(self):
        item = self.videoList.currentItem().text()
        self.filename = self.shotPath + '/' + item
        self.playcapture = cv.VideoCapture(str(self.filename ))
        self.PlayVideo()
        self.status = 0
        self.VideoPlayPause()

    def SelectPath(self):
        self.shotPath = QFileDialog.getExistingDirectory(None, "Select a directory", "../../")
        shots_files = os.listdir(self.shotPath)
        shots_files.sort()
        self.videoList.addItems(shots_files)

    def VideoPlayPause(self):
        self.status, playstr = ((1, 'pause'), (0, 'play'), (1, 'pause'))[self.status]  # 三种状态分别对应的显示、处理
        self.playbtn.setText(playstr)
        if self.status is 1:  # 状态1，播放视频
            self.timer.stop()
            self.playtimer.start()
        else:
            self.playtimer.stop()


class Timer(QThread):
    def __init__(self, signal="updateTime()", parent=None):
        super(Timer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            if self.stoped:
                return

            self.emit(SIGNAL(self.signal))
            time.sleep(0.04)  # 40毫秒发送一次信号，每秒25帧

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):
        with QMutexLocker(self.mutex):
            return self.stoped


# 弹出对话框
# add
class LabelDlg(QDialog):
    def __init__(self, title, fruit=None, parent=None):
        super(LabelDlg, self).__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        label_0 = QLabel(title)
        # 让标签字换行
        label_0.setWordWrap(True)
        self.fruit_edit = QLineEdit(fruit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        validator = QRegExp(r'[^\s][\w\s]+')
        self.fruit_edit.setValidator(QRegExpValidator(validator, self))

        v_box = QVBoxLayout()
        v_box.addWidget(label_0)
        v_box.addWidget(self.fruit_edit)
        v_box.addWidget(btns)
        self.setLayout(v_box)

        self.fruit = None

    def accept(self):
        # OK按钮
        self.fruit = unicode(self.fruit_edit.text())
        # self.done(0)
        QDialog.accept(self)

    def reject(self):
        # self.done(1)
        QDialog.reject(self)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()

    sys.exit(app.exec_())


