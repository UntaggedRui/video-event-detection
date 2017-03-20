#! /usr/bin/env python
# coding=utf-8
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import os
import cv2
import numpy as np
import time
import shutil
import subprocess

sys.path.append('../../caffe/python')
import caffe

class MainWindow(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)
        self.resize(550, 360)
        self.setWindowTitle('video-splitter')
        self.status = 0  # 0 is init status;1 is play video; 2 is capture video
        self.image = QImage()

        # 初始化按钮
        self.selectbtn = QPushButton('select')
        self.playbtn = QPushButton('play')
        self.setStartBtn = QPushButton('set start')
        self.setEndBtn = QPushButton('set end')
        self.genShotsBtn = QPushButton('generate shots')
        exitbtn = QPushButton('exit')

        self.foldernameEdit = QLineEdit('Folder name here.');

        self.piclabel = QLabel('pic')
        self.piclabel.setFixedSize(640, 360)
        self.piclabel.setScaledContents(True)
        self.slider = progressSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.current_frame = 0
        self.infobox = QTextEdit('Info here.')
        self.infobox.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)


        # 界面布局
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.piclabel)
        vbox1.addWidget(self.slider)
        vbox1.addWidget(self.infobox)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.selectbtn)
        vbox2.addWidget(self.playbtn)
        vbox2.addWidget(self.setStartBtn)
        vbox2.addWidget(self.setEndBtn)
        vbox2.addWidget(self.genShotsBtn)
        vbox2.addWidget(exitbtn)
        vbox2.addWidget(self.foldernameEdit)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        self.setLayout(hbox)

        # 设定定时器
        self.playtimer = Timer("updatePlay()")  # 播放视频

        self.genThread = GenerateShotsThread()

        # 信号--槽
        self.connect(self.playtimer, SIGNAL("updatePlay()"),
                     self.PlayVideo)
        self.connect(self.selectbtn, SIGNAL("clicked()"),
                     self.SelectVideo)
        self.connect(self.playbtn, SIGNAL("clicked()"),
                     self.VideoPlayPause)
        self.connect(self.setStartBtn, SIGNAL("clicked()"),
                     self.SetStart)
        self.connect(self.setEndBtn, SIGNAL("clicked()"),
                     self.SetEnd)
        self.connect(self.genShotsBtn, SIGNAL("clicked()"),
                     self.GenerateShots)
        self.connect(self.slider, SIGNAL("sliderMoved(int)"),
                     self.SetVideoPosition)
        self.connect(exitbtn, SIGNAL("clicked()"),
                     app, SLOT("quit()"))

        self.connect(self.genThread, SIGNAL("appendNewInfo(QString)"),
                     self.UpdateInfoBox)

        if os.path.exists('./_temp') == False:
            os.mkdir('./_temp')

    def PlayVideo(self):
        stat, frame = self.playcapture.read()
        if stat == False or self.current_frame >= self.end_frame:
            #re-play from start of the video
            self.playcapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,self.start_frame)
            self.current_frame = self.start_frame
            #self.playtimer.stop()
        else:
            self.current_frame = self.current_frame + 1
            self.slider.setValue(self.current_frame)
            frame = cv2.cv.fromarray(frame)
            cv2.cv.SaveImage('./_temp/play_temp.jpg', frame)
            self.image.load("./_temp/play_temp.jpg")
            self.piclabel.setPixmap(QPixmap.fromImage(self.image))

    def SelectVideo(self):
        self.videofile = QFileDialog.getOpenFileName(None, "Select a video file", "../../")
        self.playcapture = cv2.VideoCapture(str(self.videofile))

        self.playtimer.frame_interval = 1/self.playcapture.get(cv2.cv.CV_CAP_PROP_FPS)

        self.slider.setEnabled(True)
        self.slider_max = self.playcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.slider.setRange(0, self.slider_max)

        self.start_frame = 0
        self.end_frame = self.playcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

        self.PlayVideo()
        self.status = 0
        self.VideoPlayPause()

    def SetStart(self):
        self.start_frame = self.playcapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        self.infobox.append(QString('set start frame at %1').arg(self.start_frame))

    def SetEnd(self):
        self.end_frame = self.playcapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        self.infobox.append(QString('set end frame at %1').arg(self.end_frame))

    def VideoPlayPause(self):
        self.status, playstr = ((1, 'pause'), (0, 'play'), (1, 'pause'))[self.status]
        self.playbtn.setText(playstr)
        if self.status is 1:  # 状态1，播放视频
            self.playtimer.start()
        else:
            self.playtimer.stop()

    def SetVideoPosition(self,pos):
        self.playcapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,pos)
        self.current_frame = pos

    def GenerateShots(self):
        folder_name = self.foldernameEdit.text()
        if folder_name == 'Folder name here.':
            QMessageBox.warning(self, 'Warning', 'Please input folder name', QMessageBox.Ok)
            return

        # QString to string
        self.genThread.shot_path = unicode(folder_name.toUtf8(), 'utf-8', 'ignore') + '_shots/'
        self.genThread.thread_playcapture = self.playcapture
        self.genThread.start_frame = self.start_frame
        self.genThread.end_frame = self.end_frame
        self.genThread.start()

        # self.foldernameEdit.setText('Folder name here.')

    def UpdateInfoBox(self, info):
        self.infobox.append(info)
        cursor = self.infobox.textCursor()
        self.infobox.moveCursor(QTextCursor.End)


class Timer(QThread):
    def __init__(self, signal="updateTime()", parent=None):
        super(Timer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()
        #default fps is 25, so frame_interval is 0.04s
        self.frame_interval = 0.04

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            if self.stoped:
                return

            self.emit(SIGNAL(self.signal))
            time.sleep(self.frame_interval)  # 40毫秒发送一次信号，每秒25帧

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):
        with QMutexLocker(self.mutex):
            return self.stoped

class GenerateShotsThread(QThread):
    def __init__(self, parent = None):
        QThread.__init__(self, parent)
        self.frame_path = '_frames/'
        self.shot_path = ''
        self.thread_playcapture = None
        self.start_frame = 0
        self.stop_frame = 0

    # An individual thread used to generate frames and shots.
    def run(self):
        # split into frames
        if os.path.exists(self.frame_path) == True:
            shutil.rmtree(self.frame_path)
        os.mkdir(self.frame_path)

        self.thread_playcapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.start_frame)

        ret, frame = self.thread_playcapture.read()

        interval = 2  # interval to fetch frames
        count = self.start_frame
        index = 1  # image file index

        while count <= self.end_frame:
            ret, frame = self.thread_playcapture.read()
            count = count + 1
            if count % interval != 0:
                continue
            path = "%s%06d.jpg" % (self.frame_path, index)
            # path = save_path + bytes(index) + '.jpg';
            index = index + 1
            cv2.imwrite(path, frame)
            info = QString('split frame %1').arg(count)
            print(info)
            self.emit(SIGNAL("appendNewInfo(QString)"), info)

        # generate shots
        # model settings
        model_root = '../../models/'
        opt = {
            "debug": False,
            "caffeMode": "gpu",
            "batchSize": 1,
            "inputSize": 227,
            "net": "alexNetPlaces",
            "layer": "single",  # multi, single
            "dataset": "MITIndoor67"
        }
        caffe.set_device(0)
        caffe.set_mode_gpu()

        if opt["net"] == "googleNet":
            model_def = model_root + 'deploy_googlenet.prototxt'
            model_weights = model_root + 'imagenet_googlelet_train_iter_120000.caffemodel'
        elif opt["net"] == "alexNetPlaces":
            model_def = model_root + 'alexnet_places/places205CNN_deploy.prototxt'
            model_weights = model_root + 'alexnet_places/places205CNN_iter_300000.caffemodel'
            layer_names = ["fc7"]
        else:
            print "[Error]no model exist."
            exit()

        # net initialization
        net = caffe.Net(model_def,  # 定义模型结构
                        model_weights,  # 预训练的网络
                        caffe.TEST)  # 测试模式

        net.blobs['data'].reshape(opt["batchSize"],  # batch size
                                  3,  # BGR
                                  opt["inputSize"], opt["inputSize"])  # image size

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2, 0, 1))  # 变换image矩阵，把channel放到最后一维
        transformer.set_raw_scale('data', 255)  # 从[0,1]rescale到[0,255]
        transformer.set_channel_swap('data', (2, 1, 0))  # 调整 channels from RGB to BGR

        image_files = os.listdir(self.frame_path)
        image_files.sort()

        image_count = 0
        image_head = 1
        min_count = 20

        video_index = 1
        euclidean_threshold = 7
        feature_last = []

        if os.path.exists(self.shot_path) == True:
            shutil.rmtree(self.shot_path)
        os.mkdir(self.shot_path)

        for image_name in image_files:
            image = caffe.io.load_image(self.frame_path + image_name)  # 读入图片
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()  # 这里output是CNN最后一层的输出向量
            feature = net.blobs['fc7'].data[0]  # 读取fc7层的特征
            feature_standarlized = (feature - min(feature)) / (max(feature) - min(feature))  # 归一化

            width = int(self.thread_playcapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
            height = int(self.thread_playcapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            if image_count > 0:
                dist = np.sqrt(np.sum(np.square(feature_standarlized - feature_last)))  # Euclidean
                # dist = np.sum(np.abs(feature_standarlized - feature_last)) #Manhatten
                if (dist >= euclidean_threshold) and image_count > min_count:
                    fps = self.thread_playcapture.get(cv2.cv.CV_CAP_PROP_FPS)
                    size = (width, height)

                    fourcc = cv2.cv.FOURCC(*'XVID')
                    v = cv2.VideoWriter('%s%03d.avi' % (self.shot_path, video_index), fourcc, 25, size)

                    index = 1
                    for i in range(image_head, image_head + image_count):
                        srcfile = '_frames/' + '%06d.jpg' % i
                        frame = cv2.imread(srcfile)
                        v.write(frame)

                    info = 'start:%d,end:%d,index:%d,distance:%f' % (
                    image_head, image_head + image_count, video_index, dist)

                    self.emit(SIGNAL("appendNewInfo(QString)"), QString(info))

                    image_count = 0
                    image_head = int(image_name[:-4])
                    video_index = video_index + 1

            image_count = image_count + 1
            feature_last = feature_standarlized




class progressSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super(progressSlider, self).__init__(orientation, parent)

    def mousePressEvent(self, event):
        #if self.topLevelWidget().mediaObj.state() != 2:
        #    return
        new = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())
        self.setValue(new)
        self.emit(SIGNAL('sliderMoved(int)'), new)

    def mouseMoveEvent(self, event):
        #if self.topLevelWidget().mediaObj.state() != 2:
        #    return
        new = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())
        self.setValue(new)
        self.emit(SIGNAL('sliderMoved(int)'), new)

    def wheelEvent(self, event):
        max = self.maximum()
        min = self.minimum()
        if event.delta() >= 120:
            #滚动3%
            new = self.value()+ max*0.03
            if new > max:
                new = max
            self.setValue(new)
            self.emit(SIGNAL('sliderMoved(int)'), new)
        elif event.delta() <= -120:
            new = self.value()- max*0.03
            if new < min:
                new = min
            self.setValue(new)
            self.emit(SIGNAL('sliderMoved(int)'), new)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()

    sys.exit(app.exec_())
