#! /usr/bin/env python
# coding=utf-8
# liutingxi 2017-3-14

import sys, string, os, shutil
import cv2

def ResizeImages(srcdir,dstdir):
    srcfiles = os.listdir(srcdir)

    for srcfile in srcfiles:
        sub_srcfile = os.path.join(srcdir, srcfile)
        sub_dstfile = os.path.join(dstdir, srcfile)
        if os.path.isdir(sub_srcfile):
            if os.path.exists(sub_dstfile) == False:
                os.mkdir(sub_dstfile)
            ResizeImages(sub_srcfile,sub_dstfile)
        else:
            im = cv2.imread(sub_srcfile)
            res = cv2.resize(im, (256,256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(sub_dstfile,res)

def Rerange_Index(dir,index_start):
    temp_dir = dir + '_temp'
    if os.path.exists(temp_dir) == True:
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    files = os.listdir(dir)
    files.sort(key= lambda x:int(x[:-4]))
    index = index_start
    for srcfile in files:
        srcfile = os.path.join(dir,srcfile)
        newfile = temp_dir + "/" + "%04d"%(index) + ".jpg"
        shutil.copy(srcfile, newfile)
        index += 1
    shutil.rmtree(dir)
    shutil.move(temp_dir,dir)




if __name__ == '__main__':
    # srcdir_root = '/media/liutingxi/9C88E5A788E5805E/Event_Dataset'
    # dstdir_root = '/media/liutingxi/9C88E5A788E5805E/Event_Dataset_256'
    # if os.path.exists(dstdir_root) == False:
    #     os.mkdir(dstdir_root)
    # ResizeImages(srcdir_root, dstdir_root)

    dir = '/home/liutingxi/Research/VED/CloseupView'
    index_start = 1
    Rerange_Index(dir,index_start)
